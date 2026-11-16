"""Protocol v2: run verified OpenPhiSolve QIHD jobs sequentially on the CPU.

Arrays use NPY with allow_pickle=False. CSR indices and indptr are zero based.
The response TOML is the batch commit record; it is published only after every
job succeeds. Source hashes normalize CRLF/CR to LF and preserve all other bytes.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import tempfile
import time
import tomllib


SOURCE_REVISION = "153eb6b385a906b13807c9a28598aa44973b6cc4"
PREFLIGHT_PROFILE = "pinned-cpu-v1"
PROFILE_VERSIONS = {"jax_version": "0.6.0", "jaxlib_version": "0.6.0", "numpy_version": "2.2.6"}
BETA_SOURCE_REVISION = "private-beta-20261115221654"
BETA_PREFLIGHT_PROFILE = "private-beta-cpu-v1"
BETA_PROFILE_VERSIONS = {"jax_version": "0.12.0.1", "jaxlib_version": "0.12.0.1",
                         "numpy_version": "2.5.4", "scipy_version": "1.18.1"}
AUTO_PREFLIGHT_PROFILE = "verified-cpu-v1"
MANIFEST_TAG = b"OpenPhiSolve source manifest v1\n"
DEFAULTS = {
    "n_shots": 100, "n_steps": 10000, "ballistic": False, "verbose": False,
    "dt": 0.2, "a0": 1.0, "symplectic_integration": True, "slow_a": True,
    "lc_pr": 3.0, "constant_cons": False, "c0_multiplier": 0.5,
}


def _verify_source_manifest(filename, revision):
    """Verify the exact installed Python source set before importing phisolve."""
    with Path(__file__).with_name(filename).open("rb") as stream:
        manifest = tomllib.load(stream)
    if (manifest["manifest_version"] != 1
            or manifest["source_revision"] != revision
            or manifest["normalization"] != "CRLF/CR to LF; preserve all other bytes"):
        raise ValueError("Unsupported OpenPhiSolve source manifest.")
    expected = manifest["files"]
    spec = importlib.util.find_spec("phisolve")
    if spec is None or spec.origin is None or spec.submodule_search_locations is None:
        raise ModuleNotFoundError("Cannot locate the phisolve package that Python would import.")
    package = Path(spec.origin).resolve().parent
    locations = {Path(path).resolve() for path in spec.submodule_search_locations}
    if locations != {package} or Path(spec.origin).name != "__init__.py":
        raise ValueError("OpenPhiSolve must resolve to one regular Python package.")
    actual = {path.relative_to(package).as_posix(): path for path in package.rglob("*.py")}
    if set(actual) != set(expected) or len(actual) != manifest["file_count"]:
        raise ValueError("Installed OpenPhiSolve Python source file set differs from the manifest.")
    digest = hashlib.sha256(MANIFEST_TAG + revision.encode("ascii") + b"\n")
    for name in sorted(expected):
        normalized = actual[name].read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        checksum = hashlib.sha256(normalized).hexdigest()
        if checksum != expected[name]:
            raise ValueError(f"Installed OpenPhiSolve source differs from the manifest: {name}")
        digest.update(name.encode("utf-8") + b"\0" + checksum.encode("ascii") + b"\n")
    checksum = digest.hexdigest()
    if checksum != manifest["source_sha256"]:
        raise ValueError("OpenPhiSolve aggregate source digest differs from the manifest.")
    return checksum


def verify_source():
    """Verify the legacy installed source profile."""
    return _verify_source_manifest("source-manifest.toml", SOURCE_REVISION)


def verify_source_profile():
    """Select a supported runtime by complete source hashes, never its version label."""
    failures = []
    for filename, revision, profile in (
        ("source-manifest.toml", SOURCE_REVISION, PREFLIGHT_PROFILE),
        ("source-manifest-private-beta.toml", BETA_SOURCE_REVISION, BETA_PREFLIGHT_PROFILE),
    ):
        try:
            checksum = _verify_source_manifest(filename, revision)
        except ValueError as error:
            failures.append(f"{revision}: {error}")
            continue
        return {"source_revision": revision, "source_sha256": checksum,
                "preflight_profile": profile}
    raise ValueError("Installed OpenPhiSolve source does not match a supported manifest. "
                     + " ".join(failures))


def integer(value, name, minimum=0):
    if type(value) is not int or not minimum <= value <= (1 << 63) - 1:
        raise ValueError(f"{name} must be an integer in [{minimum}, 2^63-1].")
    return value


def job_settings(job):
    allowed = set(DEFAULTS) | {"id", "seed", "storage", "n_binary_vars", "x64"}
    if set(job) - allowed:
        raise ValueError(f"Unknown job settings: {sorted(set(job) - allowed)}")
    integer(job["id"], "id", 1)
    integer(job["n_binary_vars"], "n_binary_vars")
    settings = {name: job.get(name, default) for name, default in DEFAULTS.items()}
    settings["seed"] = integer(job["seed"], "seed")
    integer(settings["n_shots"], "n_shots", 1)
    integer(settings["n_steps"], "n_steps", 1)
    for name in ("ballistic", "verbose", "symplectic_integration", "slow_a", "constant_cons"):
        if type(settings[name]) is not bool:
            raise ValueError(f"{name} must be a boolean.")
    for name in ("dt", "a0", "lc_pr", "c0_multiplier"):
        value = settings[name]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite and real.")
        if value < 0 or (name != "lc_pr" and value == 0):
            raise ValueError(f"{name} must be positive (lc_pr may be zero).")
    if settings["slow_a"] and settings["n_steps"] <= 1000:
        raise ValueError("slow_a=true requires n_steps > 1000.")
    if type(job.get("x64", False)) is not bool:
        raise ValueError("x64 must be a boolean.")
    if job.get("storage", "dense") not in ("dense", "csr"):
        raise ValueError("storage must be 'dense' or 'csr'.")
    return settings


def load_array(path, np, *, index=False):
    array = np.load(path, allow_pickle=False)
    if not isinstance(array, np.ndarray) or array.dtype.kind not in ("iu" if index else "fiu"):
        raise ValueError(f"{path.name} must be a real numeric NPY array.")
    if not np.isfinite(array).all():
        raise ValueError(f"{path.name} contains nonfinite coefficients.")
    return array


def load_matrix(directory, name, storage, np, sparse):
    if storage == "dense":
        array = load_array(directory / f"{name}.npy", np)
        if array.ndim != 2:
            raise ValueError(f"{name} must be a matrix.")
        return array.astype(np.float64, copy=False)
    data = load_array(directory / f"{name}.data.npy", np)
    indices = load_array(directory / f"{name}.indices.npy", np, index=True)
    indptr = load_array(directory / f"{name}.indptr.npy", np, index=True)
    shape = load_array(directory / f"{name}.shape.npy", np, index=True)
    if shape.shape != (2,) or np.any(shape < 0):
        raise ValueError(f"{name} CSR shape must contain two nonnegative dimensions.")
    rows, columns = map(int, shape)
    if (data.ndim != 1 or indices.ndim != 1 or indptr.shape != (rows + 1,)
            or len(data) != len(indices) or indptr[0] != 0 or indptr[-1] != len(data)
            or np.any(indptr[1:] < indptr[:-1]) or np.any(indices < 0)
            or np.any(indices >= columns)):
        raise ValueError(f"{name} has invalid zero-based CSR components.")
    matrix = sparse.csr_matrix(
        (data.astype(np.float64), indices.astype(np.int64), indptr.astype(np.int64)),
        shape=(rows, columns),
    )
    matrix.sum_duplicates()
    matrix.sort_indices()
    matrix.eliminate_zeros()
    return matrix


def load_problem(directory, job, np, sparse, MIQP, effective_dtype):
    storage = job.get("storage", "dense")
    matrices = {name: load_matrix(directory, name, storage, np, sparse) for name in ("Q", "A", "C")}
    vectors = {name: load_array(directory / f"{name}.npy", np).astype(np.float64, copy=False)
               for name in ("w", "b", "d")}
    Q, A, C = (matrices[name] for name in ("Q", "A", "C"))
    w, b, d = (vectors[name] for name in ("w", "b", "d"))
    size = w.size
    if (size == 0 or w.shape != (size,) or Q.shape != (size, size)
            or A.shape != (b.size, size) or C.shape != (d.size, size)
            or b.ndim != 1 or d.ndim != 1 or job["n_binary_vars"] > size):
        raise ValueError("Q,w,A,b,C,d or n_binary_vars have inconsistent dimensions.")
    symmetric = (Q != Q.T).nnz == 0 if sparse.issparse(Q) else np.array_equal(Q, Q.T)
    if not symmetric:
        raise ValueError("Q must be symmetric; the worker does not symmetrize it.")
    if np.any(Q.diagonal()[:job["n_binary_vars"]] != 0):
        raise ValueError("Binary coordinates must have zero Q diagonal; fold it into constants first.")
    # Check effective JAX precision without converting a sparse matrix to dense.
    for value in (*matrices.values(), *vectors.values()):
        values = value.data if sparse.issparse(value) else value
        with np.errstate(over="ignore", invalid="ignore"):
            finite = np.isfinite(values.astype(effective_dtype)).all()
        if not finite:
            raise ValueError("Coefficients are not finite in the effective JAX dtype.")
    return MIQP(**matrices, **vectors, bounds=(-np.ones(size), np.ones(size)),
                n_binary_vars=job["n_binary_vars"])


def toml_value(value):
    if type(value) is bool:
        return str(value).lower()
    if isinstance(value, str):
        # TOML accepts literal Unicode but rejects JSON-style surrogate pairs.
        return json.dumps(value, ensure_ascii=False).replace("\x7f", "\\u007F")
    if type(value) in (int, float) and math.isfinite(value):
        return repr(value)
    raise TypeError(f"Unsupported response value: {type(value).__name__}")


def verify_consumption_profile(metadata, jax, np, dtype):
    """The float32 schedule and CPU flush-to-zero rules are runtime-specific."""
    for key, expected in PROFILE_VERSIONS.items():
        if metadata[key] != expected:
            raise ValueError(f"Unverified QIHD consumption profile: {key}={metadata[key]}, expected {expected}.")
    product = jax.jit(lambda x, y: x * y)
    for precision in {np.float32, dtype}:
        floor = np.finfo(precision).tiny
        boundary = np.array([floor / 2, floor], dtype=precision)
        observed = np.asarray(product(boundary, np.ones_like(boundary)))
        if observed.dtype != precision or not np.array_equal(observed, [0, floor]):
            raise ValueError(f"Unverified QIHD CPU arithmetic at the {precision.__name__} normality floor.")

    # Probe the source expression's operand, intermediate, potential and gradient
    # widths, independently of the dtype of the final state.
    a = jax.numpy.asarray(np.float32(0))
    state = jax.numpy.ones(1, dtype=dtype)
    def potential(x, schedule_value):
        return 0.5 * (1.0 - schedule_value) * jax.numpy.sum(x**2)
    delta = jax.eval_shape(lambda value: 1.0 - value, a)
    energy, gradient = jax.eval_shape(jax.value_and_grad(potential), state, a)
    if a.dtype != np.float32 or delta.dtype != np.float32 or energy.dtype != dtype or gradient.dtype != dtype:
        raise ValueError("Unverified QIHD schedule, binary-potential, or gradient dtype.")
    return {"admission_profile": PREFLIGHT_PROFILE, "execution_dtype": dtype.__name__,
            "delta_a_dtype": str(delta.dtype), "binary_potential_dtype": str(energy.dtype),
            "binary_gradient_dtype": str(gradient.dtype)}


def verify_beta_consumption_profile(metadata, jax, np, dtype):
    """Verify the beta's uniform execution precision and its validated CPU stack."""
    for key, expected in BETA_PROFILE_VERSIONS.items():
        if metadata[key] != expected:
            raise ValueError(f"Unverified QIHD consumption profile: {key}={metadata[key]}, expected {expected}.")
    product = jax.jit(lambda x, y: x * y)
    floor = np.finfo(dtype).tiny
    boundary = np.array([floor / 2, floor], dtype=dtype)
    observed = np.asarray(product(boundary, np.ones_like(boundary)))
    if observed.dtype != dtype or not np.array_equal(observed, [0, floor]):
        raise ValueError(f"Unverified QIHD CPU arithmetic at the {dtype.__name__} normality floor.")
    a0 = jax.numpy.asarray(dtype(1))
    a = jax.numpy.asarray(dtype(0))
    state = jax.numpy.ones(1, dtype=dtype)

    def potential(x, initial_value, schedule_value):
        return 0.5 * (initial_value - schedule_value) * jax.numpy.sum(x**2)

    delta = jax.eval_shape(lambda initial, value: initial - value, a0, a)
    energy, gradient = jax.eval_shape(jax.value_and_grad(potential), state, a0, a)
    if any(value.dtype != dtype for value in (a0, a, delta, energy, gradient)):
        raise ValueError("Unverified beta QIHD schedule, binary-potential, or gradient dtype.")
    return {"admission_profile": BETA_PREFLIGHT_PROFILE, "execution_dtype": dtype.__name__,
            "delta_a_dtype": str(delta.dtype), "binary_potential_dtype": str(energy.dtype),
            "binary_gradient_dtype": str(gradient.dtype)}


def positive_scalar(value, name, dtype, np):
    """Cast at the point of consumption and enforce the verified CPU normality floor."""
    effective = dtype(value)
    if not np.isfinite(value) or not np.isfinite(effective) or effective < np.finfo(dtype).tiny:
        raise ValueError(f"{name} must remain finite, positive and normal at {dtype.__name__} consumption "
                         f"(host={value}, cast={effective}).")
    return float(effective)


def runtime_preflight(problem, sampler, np, jax, dtype, norms):
    from phisolve.utils.jax_utils import mat_square

    diagnostics = {"effective_dt": positive_scalar(sampler.dt, "dt", dtype, np),
                   "effective_a0": positive_scalar(sampler.a0, "a0", dtype, np),
                   "schedule_a0": positive_scalar(sampler.a0, "a0 schedule", np.float32, np)}
    # Mirror c0_default's host reduction, including sparse elementwise squares.
    Q, w, A, b, C, d = (getattr(problem, key) for key in ("Q", "w", "A", "b", "C", "d"))
    squares = (mat_square(Q).sum() + (w**2).sum() + mat_square(A).sum() + np.sum(b**2)
               + mat_square(C).sum() + np.sum(d**2))
    n = w.size
    rms = np.sqrt(squares / ((n + A.shape[0] + C.shape[0]) * (n + 1)))
    if not np.isfinite(rms) or rms <= 0:
        raise ValueError(f"QIHD normalization RMS must be finite and positive in host arithmetic (R={rms}).")
    c0 = sampler.c0_multiplier / (rms * np.sqrt(n))
    diagnostics.update(normalization_rms=float(rms), c0=float(c0),
                       effective_c0=positive_scalar(c0, "c0", dtype, np))

    numerator = max(norms["Q_inf_norm"], norms["w_inf_norm"])
    for matrix, rhs in (("A", "b"), ("C", "d")):
        name = "eta_" + matrix
        denominator = max(norms[matrix + "_inf_norm"], norms[rhs + "_inf_norm"])
        # Preserve (lc_pr * numerator) / denominator in host precision. Neither
        # host control has an independent float32 cast in this part of the source.
        scale = (sampler.lc_pr * numerator) / denominator if denominator > 0 else 1.0
        if denominator > 0 and (sampler.lc_pr == 0 or numerator == 0):
            effective = 0.0
            diagnostics[name + "_zero_reason"] = "lc_pr_zero" if sampler.lc_pr == 0 else "objective_norm_zero"
        else:
            effective = positive_scalar(scale, name, dtype, np)
        diagnostics[name] = float(scale)
        diagnostics["effective_" + name] = effective

    denominator = sampler.n_steps - 1000 if sampler.slow_a else 1000
    divided = np.arange(sampler.n_steps, dtype=np.float32) / denominator
    schedule = np.where(divided > sampler.a0, sampler.a0, divided)
    transferred = jax.numpy.asarray(schedule)
    if divided.dtype != np.float32 or schedule.dtype != np.float32 or transferred.dtype != np.float32:
        raise ValueError("Unverified QIHD schedule promotion or transfer dtype.")
    if not np.isfinite(schedule).all() or schedule[0] != 0 or np.any(schedule[1:] < np.finfo(np.float32).tiny):
        raise ValueError("QIHD schedule lost a positive value or contains an invalid value.")
    diagnostics.update(schedule_division_dtype=str(divided.dtype), schedule_pretransfer_dtype=str(schedule.dtype),
                       schedule_execution_dtype=str(transferred.dtype), schedule_max=float(schedule.max()))
    return diagnostics


def beta_runtime_preflight(problem, sampler, np, jax, dtype, norms):
    """Mirror the verified beta's actual host scales and execution boundaries.

    Its source-owned checks cover every coefficient and the schedule. The
    normalization uses scaled squares, and both matrix representations use
    the maximum absolute entry. No legacy float32 schedule is imposed.
    """
    from scipy import sparse
    from phisolve.backends.qihd import _validate_execution_values
    from phisolve.utils.model_validation import validate_qihd_model
    from phisolve.exceptions import QIHDUnsupportedModelError

    validate_qihd_model(problem)
    diagnostics = {"matrix_inf_norm_convention": "maximum_absolute_entry"}
    for name in ("dt", "a0", "lc_pr", "c0_multiplier"):
        diagnostics["effective_" + name] = positive_scalar(getattr(sampler, name), name, dtype, np)
    values = [getattr(problem, name) for name in ("Q", "w", "A", "b", "C", "d")]
    for name, value in zip(("Q", "w", "A", "b", "C", "d"), values):
        _validate_execution_values(value.data if sparse.issparse(value) else value, name, dtype)

    schedule = sampler.build_schedule()
    transferred = jax.numpy.asarray(schedule)
    if schedule.dtype != dtype or transferred.dtype != dtype:
        raise ValueError("Unverified beta QIHD schedule promotion or transfer dtype.")
    diagnostics.update(schedule_a0=float(dtype(sampler.a0)),
                       schedule_division_dtype=str(schedule.dtype),
                       schedule_pretransfer_dtype=str(schedule.dtype),
                       schedule_execution_dtype=str(transferred.dtype),
                       schedule_max=float(schedule.max()))

    magnitude = max(norms.values())
    normalized_sum = 0.0
    if magnitude:
        for value in values:
            data = value.data if sparse.issparse(value) else value
            normalized = np.asarray(data, dtype=np.float64) / magnitude
            normalized_sum += np.sum(normalized * normalized, dtype=np.float64)
    n = problem.nvar
    scaled_rms = np.sqrt(normalized_sum / ((n + problem.A.shape[0] + problem.C.shape[0]) * (n + 1)))
    diagnostics.update(normalization_magnitude=float(magnitude),
                       normalization_scaled_rms=float(scaled_rms),
                       normalization_rms=float(magnitude * scaled_rms))
    numerator = max(norms["Q_inf_norm"], norms["w_inf_norm"])
    if numerator == 0:
        if norms["A_inf_norm"] > 0 or norms["C_inf_norm"] > 0:
            raise QIHDUnsupportedModelError(
                "Constraint-only QIHD requires a separately defined feasibility penalty; "
                "the legacy objective-based penalty scale is zero")
        if np.any(problem.b < 0) or np.any(problem.d != 0):
            raise ValueError("Zero-model constant linear constraints are infeasible")
        diagnostics.update(trivial_model=True, c0=0.0, effective_c0=0.0)
        return diagnostics

    c0 = sampler.c0_default(*values)
    diagnostics.update(c0=float(c0), effective_c0=positive_scalar(c0, "c0", dtype, np))
    for matrix, rhs in (("A", "b"), ("C", "d")):
        name = "eta_" + matrix
        denominator = max(norms[matrix + "_inf_norm"], norms[rhs + "_inf_norm"])
        # Keep the beta's multiply-then-divide host order before narrowing.
        scale = (sampler.lc_pr * numerator) / denominator if denominator > 0 else 1.0
        diagnostics[name] = float(scale)
        diagnostics["effective_" + name] = positive_scalar(scale, name, dtype, np)
    return diagnostics


def response_text(source_hash, results, batch_metadata=None, *, source_revision=SOURCE_REVISION):
    lines = ["protocol_version = 2", f'source_revision = "{source_revision}"',
             f'source_sha256 = "{source_hash}"']
    lines.extend(f"{key} = {toml_value(value)}" for key, value in (batch_metadata or {}).items())
    for result in results:
        lines.append("\n[[jobs]]")
        for key, value in result.items():
            if not isinstance(value, dict):
                lines.append(f"{key} = {toml_value(value)}")
        for table in ("settings", "diagnostics", "timings"):
            lines.append(f"[jobs.{table}]")
            lines.extend(f"{key} = {toml_value(value)}" for key, value in result[table].items())
    return "\n".join(lines) + "\n"


def sample(request_path: Path, output_path: Path, *, preflight=None) -> None:
    with request_path.open("rb") as stream:
        request = tomllib.load(stream)
    if (type(request.get("protocol_version")) is not int
            or request["protocol_version"] != 2 or set(request) != {"protocol_version", "jobs"}):
        raise ValueError("Expected protocol_version=2 and a jobs array.")
    jobs = request["jobs"]
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("At least one job is required.")
    settings = [job_settings(job) for job in jobs]
    if len({job["id"] for job in jobs}) != len(jobs):
        raise ValueError("Job ids must be unique.")
    x64 = jobs[0].get("x64", False)
    if any(job.get("x64", False) != x64 for job in jobs):
        raise ValueError("All jobs in one worker process must use the same x64 setting.")
    root = request_path.resolve().parent
    destinations = [root / f'job-{job["id"]}' / "samples.npy" for job in jobs]
    if output_path.exists() or any(path.exists() for path in destinations):
        raise FileExistsError("Response and samples output paths must not already exist.")

    setup_started = time.perf_counter()
    source = verify_source_profile()
    source_hash, source_revision = source["source_sha256"], source["source_revision"]
    beta = source["preflight_profile"] == BETA_PREFLIGHT_PROFILE
    if preflight is not None and preflight not in (AUTO_PREFLIGHT_PROFILE, source["preflight_profile"]):
        raise ValueError(f"Requested preflight profile {preflight!r} does not match "
                         f"verified source profile {source['preflight_profile']!r}.")
    os.environ["JAX_PLATFORMS"] = "cpu"
    import jax
    import numpy as np
    from scipy import sparse
    import phisolve
    from phisolve import MIQP, QIHD
    from phisolve.utils.jax_utils import mat_norm
    jax.config.update("jax_enable_x64", x64)
    if jax.default_backend() != "cpu":
        raise RuntimeError("Failed to select JAX's CPU backend.")
    batch_metadata = {"source_import_path": str(Path(phisolve.__file__).resolve()),
                      "python_version": platform.python_version()}
    batch_metadata.update({name.replace("-", "_") + "_version": importlib.metadata.version(name)
                           for name in ("phisolve", "jax", "jaxlib", "numpy", "scipy")})
    dtype = np.float64 if x64 else np.float32
    profile = {}
    if preflight is not None:
        verify_profile = verify_beta_consumption_profile if beta else verify_consumption_profile
        profile = verify_profile(batch_metadata, jax, np, dtype)
    batch_metadata["verified_import_setup_seconds"] = time.perf_counter() - setup_started
    results, published, staged = [], [], []
    response_part = None
    try:
        for job, options, destination in zip(jobs, settings, destinations):
            directory = root / f'job-{job["id"]}'
            started = time.perf_counter()
            problem = load_problem(directory, job, np, sparse, MIQP, dtype)
            sampler = QIHD(device="cpu", **options)
            with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
                diagnostics = {name + "_inf_norm": float(mat_norm(getattr(problem, name), ord=np.inf))
                               for name in ("Q", "w", "A", "b", "C", "d")}
                if beta:
                    try:
                        diagnostics.update(beta_runtime_preflight(problem, sampler, np, jax, dtype, diagnostics))
                    except ValueError as err:
                        raise ValueError(f'QIHD preflight failed for job {job["id"]}: {err}') from err
                    diagnostics.update(profile)
                elif profile:
                    try:
                        diagnostics.update(runtime_preflight(problem, sampler, np, jax, dtype, diagnostics))
                    except ValueError as err:
                        raise ValueError(f'QIHD preflight failed for job {job["id"]}: {err}') from err
                    diagnostics["matrix_inf_norm_convention"] = (
                        "maximum_absolute_row_sum" if sparse.issparse(problem.Q) else "maximum_absolute_entry"
                    )
                    diagnostics.update(profile)
                else:
                    c0 = sampler.c0_default(problem.Q, problem.w, problem.A, problem.b, problem.C, problem.d)
                    effective_c0 = np.asarray(c0, dtype=dtype)
                    if not np.isfinite(effective_c0) or effective_c0 <= 0:
                        raise ValueError("QIHD needs a finite, positive normalization scale in its effective dtype.")
                    diagnostics.update(c0=float(c0), effective_c0=float(effective_c0))
            loaded = time.perf_counter()
            raw = np.asarray(sampler.generate_samples(problem))
            sampled = time.perf_counter()
            size, binary = problem.nvar, job["n_binary_vars"]
            if (raw.shape != (options["n_shots"], size) or not np.isfinite(raw).all()
                    or np.any(raw < -1) or np.any(raw > 1)
                    or not np.isin(raw[:, :binary], [-1, 0, 1]).all()):
                raise RuntimeError(f'Invalid QIHD samples for job {job["id"]}.')
            samples = raw.astype(np.int8) if binary == size else raw
            # Existing job directories inherit the caller's ACLs. Avoid creating
            # a Python TemporaryDirectory, whose mode 0700 can block restricted
            # Windows tokens from opening files inside it.
            with tempfile.NamedTemporaryFile(prefix=".samples-", suffix=".npy",
                                             dir=directory, delete=False) as stream:
                staged.append((Path(stream.name), destination))
                np.save(stream, samples, allow_pickle=False)
            saved = time.perf_counter()
            results.append({
                "id": job["id"], "samples_path": f'job-{job["id"]}/samples.npy',
                "sample_dtype": str(samples.dtype), "backend_sample_dtype": str(raw.dtype),
                "requested_shots": options["n_shots"], "actual_shots": len(samples),
                "n_binary_vars": binary, "effective_x64": bool(jax.config.x64_enabled),
                "device": "cpu", "effective_seed": options["seed"],
                "storage": job.get("storage", "dense"),
                "constraint_matrix_form": job.get("storage", "dense"),
                "source_revision": source_revision, "source_sha256": source_hash,
                "settings": {**options, "device": "cpu", "x64": x64, "n_binary_vars": binary},
                "diagnostics": diagnostics,
                "timings": {"load_seconds": loaded-started, "sampling_seconds": sampled-loaded,
                            "save_seconds": saved-sampled},
            })
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=output_path.parent, delete=False) as stream:
            response_part = Path(stream.name)
            stream.write(response_text(source_hash, results, batch_metadata, source_revision=source_revision))
        for temporary, destination in staged:
            os.replace(temporary, destination)
            published.append(destination)
        os.replace(response_part, output_path)
        response_part = None
    except BaseException:
        for path in published:
            path.unlink(missing_ok=True)
        for path, _ in staged:
            path.unlink(missing_ok=True)
        if response_part is not None:
            response_part.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--preflight", choices=[AUTO_PREFLIGHT_PROFILE, PREFLIGHT_PROFILE, BETA_PREFLIGHT_PROFILE])
    args = parser.parse_args()
    sample(args.request, args.output, preflight=args.preflight)

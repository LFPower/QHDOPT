const PROTOCOL_VERSION = 2

"""Failure to execute or validate an isolated Python worker; captured stderr is retained."""
struct WorkerError <: Exception
    message::String
    stderr::String
end

function Base.showerror(io::IO, err::WorkerError)
    print(io, err.message)
    isempty(err.stderr) || print(io, "\nWorker stderr:\n", err.stderr)
end

function _write_csr(path, matrix)
    # CSC storage of the transpose is CSR storage of the original matrix.
    transposed = sparse(transpose(matrix))
    npzwrite(path * ".data.npy", nonzeros(transposed))
    npzwrite(path * ".indices.npy", Int64.(rowvals(transposed) .- 1))
    npzwrite(path * ".indptr.npy", Int64.(transposed.colptr .- 1))
    npzwrite(path * ".shape.npy", Int64[size(matrix, 1), size(matrix, 2)])
end

function _write_job(directory, id, job)
    problem = job.problem
    job_directory = joinpath(directory, "job-$id")
    mkdir(job_directory)
    use_csr = any(matrix -> matrix isa SparseMatrixCSC, (problem.Q, problem.A, problem.C))
    for (name, matrix) in (("Q", problem.Q), ("A", problem.A), ("C", problem.C))
        path = joinpath(job_directory, name)
        use_csr ? _write_csr(path, matrix) : npzwrite(path * ".npy", matrix)
    end
    for (name, vector) in (("w", problem.w), ("b", problem.b), ("d", problem.d))
        npzwrite(joinpath(job_directory, name * ".npy"), vector)
    end
    settings = copy(job.settings)
    settings["id"] = id
    settings["storage"] = use_csr ? "csr" : "dense"
    settings["n_binary_vars"] = problem.n_binary_vars
    return settings
end

function _force_stop(process)
    try
        process_exited(process) || kill(process, Base.SIGKILL)
    catch err
        process_exited(process) || rethrow(err)
    end
    wait(process)
end

function _run_worker(command, directory, timeout_seconds)
    stderr_path = joinpath(directory, "worker.stderr")
    stdout_path = joinpath(directory, "worker.stdout")
    process = try
        run(pipeline(command; stdout=stdout_path, stderr=stderr_path); wait=false)
    catch err
        captured = isfile(stderr_path) ? read(stderr_path, String) : ""
        throw(WorkerError("Could not start OpenPhiSolve worker: $(sprint(showerror, err))", captured))
    end
    try
        status = timedwait(() -> process_exited(process), timeout_seconds; pollint=0.05)
        if status == :timed_out
            _force_stop(process)
            throw(WorkerError("OpenPhiSolve worker timed out after $timeout_seconds seconds and was terminated", read(stderr_path, String)))
        end
        wait(process)
        captured = read(stderr_path, String)
        success(process) || throw(WorkerError("OpenPhiSolve worker exited with code $(process.exitcode)", captured))
        return captured
    finally
        # Also clean up if the Julia caller is interrupted while waiting.
        process_exited(process) ? wait(process) : _force_stop(process)
    end
end

function _read_result(directory, response, entry, id, job)
    get(entry, "id", nothing) == id || error("worker response changed job order or identifiers")
    get(entry, "actual_shots", nothing) == job.settings["n_shots"] || error("worker returned the wrong CPU shot count")
    get(entry, "effective_seed", nothing) == job.settings["seed"] || error("worker returned a different seed")
    get(entry, "effective_x64", nothing) === job.settings["x64"] || error("worker returned a different precision mode")
    get(entry, "device", nothing) == "cpu" || error("worker did not report CPU execution")
    for key in ("source_revision", "source_sha256")
        get(entry, key, nothing) == response[key] || error("worker job provenance disagrees with response provenance")
    end
    settings = get(entry, "settings", nothing)
    settings isa AbstractDict || error("worker effective settings are missing")
    for (key, value) in job.settings
        get(settings, key, nothing) == value || error("worker effective setting $key disagrees with the request")
    end
    get(settings, "n_binary_vars", nothing) == job.problem.n_binary_vars || error("worker changed the binary-variable count")
    get(settings, "device", nothing) == "cpu" || error("worker effective settings do not select CPU")
    raw_path = get(entry, "samples_path", nothing)
    raw_path isa String || error("worker response is missing samples_path")
    path = normpath(isabspath(raw_path) ? raw_path : joinpath(directory, raw_path))
    expected_path = normpath(joinpath(directory, "job-$id", "samples.npy"))
    comparison = Sys.iswindows() ? lowercase : identity
    comparison(path) == comparison(expected_path) || error("worker returned an unexpected samples path")
    samples = npzread(path)
    expected_shape = (job.settings["n_shots"], length(job.problem.w))
    samples isa AbstractMatrix{<:Real} && size(samples) == expected_shape || error("worker returned samples with an unexpected shape or type")
    all(value -> isfinite(value) && -1 <= value <= 1, samples) || error("worker returned non-finite or out-of-bounds samples")
    all(value -> value in (-1, 0, 1), @view samples[:, 1:job.problem.n_binary_vars]) || error("worker returned invalid binary spins")
    dtype = get(entry, "sample_dtype", nothing)
    expected_dtype = job.problem.n_binary_vars == length(job.problem.w) ? "int8" : (job.settings["x64"] ? "float64" : "float32")
    dtype == expected_dtype || error("worker returned an unexpected sample dtype")
    expected_type = expected_dtype == "int8" ? Int8 : (expected_dtype == "float64" ? Float64 : Float32)
    eltype(samples) === expected_type || error("samples file dtype disagrees with response metadata")
    metadata = Dict{String,Any}(key => value for (key, value) in response if key != "jobs")
    merge!(metadata, entry)
    metadata["requested_settings"] = copy(job.settings)
    # The temporary exchange files are removed after this call.
    delete!(metadata, "samples_path")
    return SampleResult(Matrix{Float64}(samples), metadata)
end

"""
    sample_jobs(jobs; python, timeout_seconds=120)

Sample multiple normalized quadratic problems in one isolated CPU worker.
Jobs share one precision mode. Results preserve job, row and column order and
retain multiplicities. Sparse matrices are transported as CSR arrays without
densifying. A bounded timeout terminates and reaps the worker. No package
installation, coordinate transform, decoding, or refinement occurs.
"""
function sample_jobs(jobs::AbstractVector{SamplingJob}; python::AbstractString, timeout_seconds::Real=120)
    return _sample_jobs(jobs; python, timeout_seconds, preflight=false)
end

function _sample_jobs(jobs::AbstractVector{SamplingJob}; python::AbstractString, timeout_seconds::Real=120,
                      preflight::Bool, remaining_time=() -> Inf)
    isempty(jobs) && throw(ArgumentError("at least one sampling job is required"))
    isempty(strip(python)) && throw(ArgumentError("python must name a Python executable"))
    !(timeout_seconds isa Bool) && isfinite(timeout_seconds) && timeout_seconds > 0 || throw(ArgumentError("timeout_seconds must be finite and positive"))
    all(job.settings["x64"] === jobs[1].settings["x64"] for job in jobs) ||
        throw(ArgumentError("all jobs in one worker must use the same x64 setting"))
    worker = normpath(joinpath(@__DIR__, "..", "python", "openphisolve_worker.py"))
    isfile(worker) || throw(ArgumentError("OpenPhiSolve worker was not found at $worker"))
    return mktempdir() do directory
        start_ns = time_ns()
        request_path, response_path = joinpath(directory, "request.toml"), joinpath(directory, "response.toml")
        request = Dict("protocol_version" => PROTOCOL_VERSION, "jobs" => [_write_job(directory, id, job) for (id, job) in enumerate(jobs)])
        open(request_path, "w") do io
            TOML.print(io, request; sorted=true)
        end
        admission_args = preflight ? ["--preflight", "verified-cpu-v1"] : String[]
        command = addenv(`$(String(python)) -I -B -X utf8 $worker $request_path $response_path $admission_args`, "JAX_PLATFORMS" => "cpu")
        serialized_ns = time_ns()
        captured_stderr = _run_worker(command, directory, min(Float64(timeout_seconds), remaining_time()))
        returned_ns = time_ns()
        try
            response = TOML.parsefile(response_path)
            get(response, "protocol_version", nothing) == PROTOCOL_VERSION || error("unsupported worker response protocol")
            get(response, "source_revision", nothing) isa String || error("worker source revision is missing")
            hash = get(response, "source_sha256", nothing)
            hash isa String && occursin(r"^[0-9a-f]{64}$", hash) || error("worker source hash is missing or malformed")
            entries = get(response, "jobs", nothing)
            entries isa Vector && length(entries) == length(jobs) || error("worker response has the wrong job count")
            results = [_read_result(directory, response, entries[id], id, job) for (id, job) in enumerate(jobs)]
            if preflight
                read_ns = time_ns()
                for result in results
                    expected_profile = get(Dict(
                        "153eb6b385a906b13807c9a28598aa44973b6cc4" => "pinned-cpu-v1",
                        "private-beta-20261115221654" => "private-beta-cpu-v1",
                    ), result.metadata["source_revision"], nothing)
                    !isnothing(expected_profile) &&
                        get(result.metadata["diagnostics"], "admission_profile", nothing) == expected_profile ||
                        error("worker did not verify the requested admission profile")
                    result.metadata["bridge_timings"] = Dict(
                        "serialization_seconds" => (serialized_ns - start_ns) / 1e9,
                        "worker_roundtrip_seconds" => (returned_ns - serialized_ns) / 1e9,
                        "readback_seconds" => (read_ns - returned_ns) / 1e9,
                    )
                end
            end
            return results
        catch err
            throw(WorkerError("Invalid OpenPhiSolve response: $(sprint(showerror, err))", captured_stderr))
        end
    end
end

"""Return samples and metadata for a single normalized `QuadraticProblem`."""
function sample_quadratic(problem::QuadraticProblem; python::AbstractString, timeout_seconds::Real=120, kwargs...)
    return only(sample_jobs([SamplingJob(problem; kwargs...)]; python, timeout_seconds))
end

"""
    sample_spins(Q, w; python, kwargs...)

Compatibility entrypoint returning only an ordered `Matrix{Float64}` for an
all-binary, unconstrained normalized spin problem. Zero spins are preserved as
invalid binary outcomes. Use `sample_quadratic` to retain metadata or to sample
mixed/continuous variables and native linear constraints.
"""
function sample_spins(Q::AbstractMatrix{<:Real}, w::AbstractVector{<:Real}; python::AbstractString, kwargs...)
    problem = QuadraticProblem(Q, w; n_binary_vars=length(w))
    return sample_quadratic(problem; python, kwargs...).samples
end

# QHDOPT public-distribution candidate

QHDOPT solves bounded continuous optimization problems using a CPU sampling worker, checks candidate points against the original bounds and objective, and can apply local refinement. Use the native Julia API or the JuMP interface.

This is a local work-in-progress candidate for a possible public distribution. It is not approved for publication: QHDOPT's project-level licensing and distribution terms remain undecided. The included third-party licenses apply to their own components and do not assign a license to QHDOPT.

Sampling is heuristic: a usable candidate is not a general proof of global optimality. Use Windows x86-64, Julia 1.13.1, Python 3.12, and the included beta CPU worker. Linux, macOS, GPU, and quantum-hardware use are not supported by this candidate.

## Install the local candidate

Obtain the complete local candidate directory from the person preparing it. Open PowerShell 7.3 or newer in the candidate checkout root, the directory containing Project.toml and this README. These instructions do not designate a public download or repository.

Install 64-bit Julia 1.13.1 and Python 3.12. Make their executables available as julia and python, or set the variables below to the corresponding installed executable paths.

~~~powershell
$julia = (Get-Command julia).Source
$basePython = (Get-Command python).Source
& $julia --version
& $basePython --version

. .\scripts\session.ps1
& $julia --startup-file=no scripts/setup_julia.jl
& $basePython python/setup_private_beta.py python/wheels/phisolve-0.3.1b20261115221654-py3-none-any.whl --venv .local/venvs/beta
~~~

The Julia setup creates .local/julia-env with QHDOPT, NPZ 0.4.3, JuMP 1.32.1, MathOptInterface 1.54.0, and Ipopt 1.16.0. It preserves this package's root Project.toml. Use this generated environment for every command below; activating only the package root does not install all optional interfaces.

The Python installer checks the included wheel's digest, creates a separate environment, installs the exact versions in python/requirements-private-beta-cpu.txt, and verifies the installed source profile. Ordinary dependencies are downloaded from their configured package sources. The pinned beta wheel is included; no private repository, author credentials, or local archive is required to obtain the worker.

The session script directs package caches, Julia's depot, temporary files, and the worker environment to ignored .local/ locations. Its settings apply to the current PowerShell process. In each new session, enter this checkout, select the Julia executable, and run:

~~~powershell
$julia = (Get-Command julia).Source
. .\scripts\session.ps1
~~~

If the checkout is moved, rerun the Julia setup and recreate the beta environment at its new location. Generated environments contain local paths and are not portable installations.

## Native Julia example

Run the included seeded example with the explicitly selected beta interpreter:

~~~powershell
& $julia --startup-file=no "--project=$QHDOPT_PROJECT" examples/native_beta.jl $QHDOPT_PYTHON
~~~

Its essential API usage is:

~~~julia
using QHDOPT

python = abspath(only(ARGS))
model = quadratic_problem([2.0 0.5; 0.5 1.0], [-1.5, 0.5];
    constant=1, lower=[0, -2], upper=[2, 1])
result = solve(model;
    encoding=Unary(2), validity=FoldedPenalty(10), storage=:dense,
    backend=PythonQIHDBackend(; python),
    sampler=QIHDConfig(seed=7, n_shots=64, n_steps=1200))
if result.selected !== nothing
    println(solution(result), " objective=", result.selected.value)
end
~~~

Use finite bounds for every variable. Equal lower and upper bounds fix a coordinate. In addition to quadratic objectives, the native structured frontend accepts its supported bounded expressions, such as:

~~~julia
model = structured_problem(:((x - 0.5)^2);
    names=[:x], lower=[0.0], upper=[1.0])
~~~

Unsupported operators or unsafe evaluation domains are rejected. This interface does not accept arbitrary nonlinear programs.

## JuMP example

~~~powershell
& $julia --startup-file=no "--project=$QHDOPT_PROJECT" examples/jump_beta.jl $QHDOPT_PYTHON
~~~

The example uses the same beta interpreter and sampler settings:

~~~julia
using JuMP
import QHDOPT

python = abspath(only(ARGS))
model = Model(() -> QHDOPT.Optimizer(
    backend=QHDOPT.PythonQIHDBackend(; python),
    encoding=QHDOPT.Unary(2), validity=QHDOPT.FoldedPenalty(10),
    sampler=QHDOPT.QIHDConfig(seed=7, n_shots=64, n_steps=1200)))
@variable(model, 0 <= x <= 2)
@variable(model, -2 <= y <= 1)
@objective(model, Min, 1 + x^2 + 0.5x*y + 0.5y^2 - 1.5x + 0.5y)
optimize!(model)
println(termination_status(model), " / ", primal_status(model))
if has_values(model)
    println([value(x), value(y)], " objective=", objective_value(model))
end
~~~

Only supported continuous, finitely bounded objectives are admitted. General high-level affine constraints, integer variables, binary variables, and nonlinear constraints are unsupported.

## Local refiners

The packaged Windows TNC library needs no local compiler. To refine sampled candidates, add refiner=TNCRefiner() to the native solve options, or refiner=QHDOPT.TNCRefiner() to the JuMP optimizer options.

Ipopt is installed by the Julia setup. Load it with using Ipopt before choosing IpoptRefiner(tol=1e-8). Using a refiner is optional; the sampler defaults are unchanged.

A small supplied-start example exercises both refiners without launching the sampling worker:

~~~powershell
& $julia --startup-file=no "--project=$QHDOPT_PROJECT" examples/refinement.jl
~~~

A refiner's termination code and whether its returned point was accepted are separate. Inspect the refinement record's status, solver_called, and usable fields when diagnosing a run. A local refiner does not establish global optimality.

## Run the basic checks

After installation, run:

~~~powershell
& $julia --startup-file=no "--project=$QHDOPT_PROJECT" test/runtests.jl $QHDOPT_PYTHON
~~~

The checks exercise module loading, the exact beta source profile, a seeded native solve, a JuMP solve and result statuses, the packaged TNC library, a structured objective, Ipopt, and representative unsupported inputs. Missing dependencies or native libraries fail the checks; required interface checks are not silently skipped. The checks do not require the stochastic sampler to find a global optimum.

The check script takes exactly one Python argument, as do the sampling examples. It does not silently select another Python environment.

## Interpret results

- An invalid model or unsupported input is rejected; a failed model is not a usable optimization result.
- In native Julia, check result.selected or solution(result) before using a point. Empty or entirely invalid samples can yield no solution.
- In JuMP, check termination_status, primal_status, result_count, and has_values before reading values.
- A sampled usable result normally reports OTHER_LIMIT with FEASIBLE_POINT. This means a checked candidate was found within the sampling budget, not that global optimality was proved.
- OPTIMAL applies to the implemented deterministic constant/all-fixed shortcuts, not ordinary sampled solutions.
- Time limits are cooperative around some Julia/refinement work; the Python worker has its separate subprocess timeout.
- A fixed seed aids repeatability within the installed CPU profile. It is not a promise of identical results across dependency profiles or platforms.

## Troubleshooting

| Symptom | Check |
|---|---|
| Julia or Python version is rejected | Select Julia 1.13.1 and a 64-bit Python 3.12 executable. |
| JuMP, Ipopt, or an extension is missing | Rerun scripts/setup_julia.jl and use --project=$QHDOPT_PROJECT. |
| Beta wheel digest or source-profile verification fails | Restore the unchanged included wheel and manifests, then install into a fresh beta environment. Do not bypass verification. |
| Wrong worker revision | Pass $QHDOPT_PYTHON explicitly. The documented installation supports the included beta profile. |
| A pinned dependency cannot be downloaded | Check the package-index/network configuration and the exact failing package/version. Do not replace pins with latest versions. |
| TNC library is missing or its digest fails | Confirm Windows x86-64 and restore the included native/tnc files. A compiler is not required for the packaged library. |
| Installation worked before the checkout moved | Recreate the local environments at the new path and rerun setup. |
| No usable candidate is returned | Inspect the status and trial counts before accessing results; sampling can return no valid point. |

## Licensing and publication status

QHDOPT's project-level licensing and distribution terms remain undecided. This local candidate is not approved for public publication and is not presented as an open-source release.

The included beta worker and native TNC files retain their component licenses. See THIRD_PARTY_NOTICES.txt, python/wheels/LICENSE.OpenPhiSolve, and the licenses alongside native/tnc source and its packaged Windows library. These component terms do not assign a license to QHDOPT. Dependencies installed separately retain their own licenses and notices.

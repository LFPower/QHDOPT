using Test
using TOML
using SHA
using JuMP
using Ipopt
import MathOptInterface as MOI
import QHDOPT

const ROOT = normpath(joinpath(@__DIR__, ".."))
length(ARGS) == 1 || error("usage: julia --startup-file=no --project=.local/julia-env test/runtests.jl <beta-python>")
const PYTHON = abspath(only(ARGS))
isfile(PYTHON) || error("The explicitly selected beta Python executable is missing.")
const REVISION = "private-beta-20261115221654"

function checked_candidate(result, lower, upper)
    @test !isnothing(result.selected)
    point = QHDOPT.solution(result)
    @test point !== nothing
    @test length(point) == length(lower)
    @test all(isfinite, point)
    @test all(lower .<= point) && all(point .<= upper)
    @test isfinite(result.selected.value)
    return point
end

function checked_batch(result, shots)
    @test result.raw !== nothing
    @test result.raw.metadata["source_revision"] == REVISION
    @test result.raw.metadata["diagnostics"]["admission_profile"] == "private-beta-cpu-v1"
    @test result.raw.metadata["device"] == "cpu"
    @test result.counts.actual_trials == shots
    @test 0 < result.counts.valid <= shots
end

@testset "QHDOPT internal-user checks" begin
    @testset "Installation and verified beta worker" begin
        @test Sys.iswindows() && Sys.ARCH == :x86_64
        @test VERSION == v"1.13.1"
        @test realpath(pathof(QHDOPT)) == realpath(joinpath(ROOT, "src", "QHDOPT.jl"))
        @test Base.get_extension(QHDOPT, :QHDOPTMathOptInterfaceExt) !== nothing
        @test Base.get_extension(QHDOPT, :QHDOPTIpoptExt) !== nothing
        worker = joinpath(ROOT, "python", "openphisolve_worker.py")
        verification = """
        import json,pathlib,runpy,sys
        import phisolve
        profile=runpy.run_path(sys.argv[1])['verify_source_profile']()
        assert profile['source_revision']=='private-beta-20261115221654',profile
        assert pathlib.Path(phisolve.__file__).resolve().is_relative_to(pathlib.Path(sys.prefix).resolve())
        print(json.dumps({'python':sys.executable,'prefix':sys.prefix,'base_prefix':sys.base_prefix,
                         'worker_package':phisolve.__file__,'profile':profile}))
        """
        @test success(Cmd([PYTHON, "-I", "-B", "-c", verification, worker]))
        manifest = TOML.parsefile(joinpath(ROOT, "native", "tnc", "source.toml"))
        for (name, expected) in manifest["sha256"]
            @test bytes2hex(sha256(read(joinpath(ROOT, "native", "tnc", name)))) == expected
        end
        println("QHDOPT loaded from: ", pathof(QHDOPT))
        println("Active Julia project: ", Base.active_project())
    end

    lower, upper = [0.0, -2.0], [2.0, 1.0]
    native = QHDOPT.quadratic_problem([2.0 0.5; 0.5 1.0], [-1.5, 0.5];
        constant=1, lower, upper)
    options = (; backend=QHDOPT.PythonQIHDBackend(python=PYTHON),
        encoding=QHDOPT.Unary(2), validity=QHDOPT.FoldedPenalty(10), storage=:dense,
        sampler=QHDOPT.QIHDConfig(seed=7, n_shots=64, n_steps=1200))

    @testset "Seeded native CPU solve" begin
        result = QHDOPT.solve(native; options...)
        checked_batch(result, 64)
        point = checked_candidate(result, lower, upper)
        x, y = point
        @test result.status == :coarse
        @test result.selected.value ≈ 1 + x^2 + 0.5x*y + 0.5y^2 - 1.5x + 0.5y atol=1e-12 rtol=1e-12
    end

    @testset "JuMP CPU solve and status interpretation" begin
        model = Model(() -> QHDOPT.Optimizer(; options..., refiner=QHDOPT.TNCRefiner()))
        @variable(model, 0 <= x <= 2)
        @variable(model, -2 <= y <= 1)
        @objective(model, Min, 1 + x^2 + 0.5x*y + 0.5y^2 - 1.5x + 0.5y)
        optimize!(model)
        @test termination_status(model) == MOI.OTHER_LIMIT
        @test primal_status(model) == MOI.FEASIBLE_POINT
        @test result_count(model) == 1
        @test has_values(model)
        result = get_attribute(model, QHDOPT.NativeResult())
        checked_batch(result, 64)
        point = checked_candidate(result, lower, upper)
        @test [value(x), value(y)] == point
        @test objective_value(model) == result.selected.value
        @test result.counts.refinement_calls > 0
        @test any(r -> r.solver_called && r.solver == :tnc && r.usable, result.refinement_records)
    end

    @testset "Packaged TNC and structured objective" begin
        model = QHDOPT.structured_problem(:((x - 0.5)^2); names=[:x], lower=[0.0], upper=[1.0])
        result = QHDOPT.solve_classical(model; starts=[[0.9]], refiner=QHDOPT.TNCRefiner())
        point = checked_candidate(result, [0.0], [1.0])
        record = only(result.refinement_records)
        @test result.counts.refinement_calls == 1
        @test record.solver_called && record.solver == :tnc && record.usable
        @test point ≈ [0.5] atol=1e-7 rtol=1e-8
        @test result.selected.value <= only(result.supplied_starts).value
        @test isfile(record.diagnostics.artifact.path)
        @test realpath(record.diagnostics.artifact.path) ==
            realpath(joinpath(ROOT, "native", "tnc", "artifacts", "windows-x86_64", "qhdopt_tnc.dll"))
        println("Loaded TNC: ", record.diagnostics.artifact.path)
    end

    @testset "Ipopt refiner" begin
        result = QHDOPT.solve_classical(native; starts=[[0.0, -2.0]],
            refiner=QHDOPT.IpoptRefiner(tol=1e-8))
        point = checked_candidate(result, lower, upper)
        record = only(result.refinement_records)
        @test result.counts.refinement_calls == 1
        @test record.solver_called && record.solver == :ipopt && record.usable
        @test record.status == :converged
        @test point ≈ [1.0, -1.0] atol=1e-5 rtol=0
        @test result.selected.value <= only(result.supplied_starts).value
    end

    @testset "Unsupported inputs and unavailable results" begin
        @test_throws ArgumentError QHDOPT.quadratic_problem(reshape([2.0], 1, 1), [-1.0]; lower=1, upper=0)
        model = Model(() -> QHDOPT.Optimizer(; options...))
        @variable(model, unbounded)
        @objective(model, Min, unbounded^2)
        optimize!(model)
        @test termination_status(model) == MOI.INVALID_MODEL
        @test result_count(model) == 0
        @test !has_values(model)
        @test get_attribute(model, QHDOPT.NativeResult()) === nothing

        unsupported = Model(() -> QHDOPT.Optimizer(; options...); add_bridges=false)
        @variable(unsupported, 0 <= z[1:2] <= 1)
        @test_throws "not supported" set_integer(z[1])
        @test_throws "not supported" begin
            @constraint(unsupported, z[1] + z[2] <= 1)
        end
    end
end

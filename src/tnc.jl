"""
    TNCRefiner(; gtol=1e-6, maxfun=nothing, maxCGit=-1, scale=nothing, offset=nothing, ...)

Minimize with the C core of SciPy 1.11.4 and supplied original-coordinate
gradients. Omitted `maxfun` is `max(100,10*n_free)`. Scale/offset vectors are
in original variable order and are copied. `library=:auto` uses the verified
packaged binary when available. `library=:local` requires an explicit local
build with `python native/tnc/build.py`. Solving never installs a library.
"""
struct TNCRefiner <: AbstractRefiner
    gtol::Float64
    maxfun::Union{Nothing,Int}
    maxCGit::Int
    eta::Float64
    stepmx::Float64
    accuracy::Float64
    minfev::Float64
    ftol::Float64
    xtol::Float64
    rescale::Float64
    scale::Union{Nothing,Tuple{Vararg{Float64}}}
    offset::Union{Nothing,Tuple{Vararg{Float64}}}
    feasibility_tolerance::Float64
    library::Symbol
    function TNCRefiner(; gtol=1e-6, maxfun=nothing, maxCGit=-1, eta=-1, stepmx=0,
                        accuracy=0, minfev=0, ftol=-1, xtol=-1, rescale=-1, scale=nothing,
                        offset=nothing, feasibility_tolerance=0.0, library=:auto)
        library in (:auto, :local) || throw(ArgumentError("TNC library must be :auto or :local"))
        if !isnothing(maxfun)
            maxfun isa Integer && !(maxfun isa Bool) && 0 <= maxfun <= typemax(Cint) ||
                throw(ArgumentError("maxfun must be a nonnegative C integer"))
        end
        maxCGit isa Integer && !(maxCGit isa Bool) && typemin(Cint) <= maxCGit <= typemax(Cint) ||
            throw(ArgumentError("maxCGit must be a C integer"))
        function scalar(x, name)
            x isa Real && !(x isa Bool) || throw(ArgumentError("TNC $name must be a finite real"))
            return _coefficient(x, "TNC $name")
        end
        function vector(v, name)
            isnothing(v) && return nothing
            v isa AbstractVector || throw(ArgumentError("TNC $name must be an original-coordinate vector"))
            return Tuple(scalar(x, name) for x in v)
        end
        feas = scalar(feasibility_tolerance, :feasibility_tolerance)
        feas >= 0 || throw(ArgumentError("feasibility_tolerance must be nonnegative"))
        scales, offsets = vector(scale,:scale), vector(offset,:offset)
        if !isnothing(scales) && any(!iszero, scales) && isnothing(offsets)
            throw(ArgumentError("SciPy 1.11.4 TNC requires an explicit offset with nonzero explicit scales to avoid uninitialized C offsets"))
        end
        return new(scalar(gtol,:gtol), maxfun, maxCGit, scalar(eta,:eta), scalar(stepmx,:stepmx),
                   scalar(accuracy,:accuracy), scalar(minfev,:minfev), scalar(ftol,:ftol),
                   scalar(xtol,:xtol), scalar(rescale,:rescale), scales, offsets, feas, library)
    end
end

const _TNC_ROOT = normpath(joinpath(@__DIR__, "..", "native", "tnc"))
const _TNC_COMMIT = "5e4a5e3785f79dd4e8930eed883da89958860db2"

function _refiner_info(r::TNCRefiner, model::QHDProblem)
    for (name, values) in ((:scale, r.scale), (:offset, r.offset))
        if !isnothing(values) && length(values) != length(model.names)
            _invalid_option(:refinement, DimensionMismatch("TNC $name must have one entry per original coordinate"))
        end
    end
    options = (; r.gtol, r.maxfun, r.maxCGit, r.eta, r.stepmx, r.accuracy, r.minfev, r.ftol,
               r.xtol, r.rescale, r.scale, r.offset, r.library, derivative=:supplied_gradient,
               messages=0, finite_difference_eps=:unused, maxfun_default=:max_100_or_10_per_free_coordinate)
    return (; solver=:tnc, options, versions=(scipy="1.11.4", commit=_TNC_COMMIT, abi=1),
            feasibility_tolerance=r.feasibility_tolerance)
end

function _tnc_installation(r; root=_TNC_ROOT)
    system = Sys.iswindows() ? "windows" : Sys.isapple() ? "darwin" : Sys.islinux() ? "linux" : ""
    isempty(system) && _invalid_option(:refinement, ArgumentError("unsupported native TNC platform $(Sys.KERNEL)"))
    platform = "$system-$(Sys.ARCH)"
    source = TOML.parsefile(joinpath(root, "source.toml"))
    source["commit"] == _TNC_COMMIT && source["abi"] == 1 || throw(ArgumentError("unexpected TNC source manifest"))
    binary = Sys.iswindows() ? "qhdopt_tnc.dll" : "libqhdopt_tnc.$(Libdl.dlext)"
    prebuilt = get(source["prebuilt"], platform, nothing)
    local_build = r.library == :local || isnothing(prebuilt)
    dir = joinpath(root, local_build ? "build" : "artifacts", platform)
    path = joinpath(dir, binary)
    isfile(path) || _invalid_option(:refinement, ArgumentError("TNC library missing at $path; run python native/tnc/build.py and select library=:local"))
    expected = if local_build
        receipt_path = joinpath(dir, "build.toml")
        isfile(receipt_path) || throw(ArgumentError("TNC build receipt missing; rerun python native/tnc/build.py"))
        receipt = TOML.parsefile(receipt_path)
        get(receipt, "platform", "") == platform && get(receipt, "commit", "") == _TNC_COMMIT &&
            get(receipt, "abi", 0) == 1 || throw(ArgumentError("TNC build receipt does not match the platform/source/ABI"))
        get(receipt, "sha256", "")
    else
        prebuilt["sha256"]
    end
    observed = open(io -> bytes2hex(sha256(io)), path)
    expected == observed || throw(ArgumentError("TNC library SHA-256 mismatch at $path; restore the packaged artifact or rebuild locally"))
    return (; path, sha256=observed, platform, origin=local_build ? :local : :packaged)
end

mutable struct _TNCState{D}
    ctx::D
    full::Vector{Float64}
    last_x::Vector{Float64}
    value::Float64
    gradient::Vector{Float64}
    nfev::Int
    callbacks::Int
    failure::Any
end

# Like SciPy's ScalarFunction, reuse an evaluation only at an identical point.
# The separate final verification below always reevaluates the returned x.
function _tnc_callback(xp::Ptr{Cdouble}, fp::Ptr{Cdouble}, gp::Ptr{Cdouble}, opaque::Ptr{Cvoid})::Cint
    state = unsafe_pointer_to_objref(opaque)::_TNCState
    try
        Base.reenable_sigint() do
            state.callbacks += 1
            x = unsafe_wrap(Vector{Float64}, xp, length(state.last_x))
            if x != state.last_x
                state.nfev += 1
                state.full[state.ctx.free] = x
                _derivative_point(state.ctx.model, state.full)
                state.value = objective_value(state.ctx.model, state.full)
                _gradient!(state.gradient, state.ctx, state.full)
                copyto!(state.last_x, x)
            end
            unsafe_store!(fp, state.value)
            unsafe_copyto!(gp, pointer(state.gradient), length(x))
        end
        return 0
    catch err
        state.failure = err
        return 1  # the C callback's documented immediate abort code
    end
end

function _tnc_check_scaling(r, state)
    # Match the order of native/tnc/tnc.c's finite-bound initialization.
    # Do not substitute a mathematically equivalent midpoint/width formula:
    # the pinned core itself would still perform the overflowing operation.
    for (k, i) in enumerate(state.ctx.free)
        lo, hi = state.ctx.model.lower[i], state.ctx.model.upper[i]
        scale = isnothing(r.scale) ? hi-lo : abs(r.scale[i])
        if iszero(scale)
            # C first fixes this variable at x but then lets an explicit
            # offset override its unscaling anchor.
            (isnothing(r.offset) || r.offset[i] == state.last_x[k]) ||
                throw(ArgumentError("TNC zero scale at coordinate $i needs its start as the offset"))
            continue
        end
        offset = isnothing(r.offset) ? (hi+lo)*0.5 : r.offset[i]
        initial = (state.last_x[k]-offset)/scale
        if !all(isfinite, (scale, offset, initial, initial*scale+offset, state.gradient[k]*scale))
            throw(ArgumentError("TNC scaling overflows at original coordinate $i; supply representable scale and offset"))
        end
    end
    return nothing
end

function _tnc_solve(func, r, model, start, ctx, artifact, budget, scale, offset, state)
    x = copy(state.last_x)
    n = length(x)
    lower, upper = model.lower[ctx.free], model.upper[ctx.free]
    f, g = Ref(Inf), zeros(n)
    nf, nit = Ref{Cint}(0), Ref{Cint}(0)
    callback = @cfunction(_tnc_callback, Cint, (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cvoid}))
    code = GC.@preserve state begin
        Base.disable_sigint() do
            ccall(func, Cint, (Cint, Ptr{Cdouble}, Ref{Cdouble}, Ptr{Cdouble}, Ptr{Cvoid}, Ptr{Cvoid},
                   Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cint, Cint, Cdouble,
                   Cdouble, Cdouble, Cdouble, Cdouble, Cdouble, Cdouble, Cdouble, Ref{Cint}, Ref{Cint}, Ptr{Cvoid}),
                  n, x, f, g, callback, pointer_from_objref(state), lower, upper, scale, offset, 0,
                  r.maxCGit, budget, r.eta, r.stepmx, r.accuracy, r.minfev, r.ftol, r.xtol, r.gtol,
                  r.rescale, nf, nit, C_NULL)
        end
    end
    native = (; native_evaluations=Int(nf[]), iterations=Int(nit[]), callback_calls=state.callbacks,
              maxfun=budget, scipy_success=0 <= code <= 2, artifact)
    result(status, diagnostics; point=nothing, message="") = RefinerOutcome(;
        status, solver_called=true, raw_status=Int(code), point, message, diagnostics)
    if !isnothing(state.failure) || code == 7
        err = state.failure
        err isa InterruptException && throw(err)  # already safely back in Julia
        status = err isa Union{ArgumentError,DomainError,OverflowError,InexactError} ? :invalid_evaluation : :callback_failure
        reason = isnothing(err) ? "TNC aborted its callbacks" : sprint(showerror, err)
        return result(status, (; native..., nfev=state.nfev, final_reevaluations=0); message=reason)
    end
    (0 <= code <= 6) || return result(:solver_failure, (; native..., nfev=state.nfev, final_reevaluations=0);
                                        message="TNC setup or memory error")
    full = copy(start.point)
    full[ctx.free] = x
    status = code <= 2 ? :converged : code == 3 ? :limit_reached : code == 5 ? :constant : :solver_failure
    nfev = state.nfev + (x != state.last_x)
    if !all(isfinite, full) || any(i -> max(model.lower[i]-full[i], full[i]-model.upper[i]) >
                                   r.feasibility_tolerance, eachindex(full))
        return result(status, (; native..., nfev, final_reevaluations=0, final_evaluation=:outside_box);
                      point=full, message="TNC returned an infeasible point; final derivatives were not evaluated")
    end
    # The C API explicitly disclaims synchronization of its x,f,g outputs.
    # Preserve raw x for the shared independent validator. Any permitted
    # roundoff projection must also be used for the reported final gradient.
    evaluated = clamp.(full, model.lower, model.upper)
    final_gradient = zeros(n)
    counts = (; native..., nfev, final_reevaluations=1, final_projected=evaluated != full)
    value = try
        v = objective_value(model, evaluated)
        _gradient!(final_gradient, ctx, evaluated)
        v
    catch err
        err isa InterruptException && rethrow()
        return result(status, (; counts..., final_evaluation=:failed); point=full,
                      message="TNC final reevaluation: $(sprint(showerror, err))")
    end
    return result(status, (; counts..., final_evaluation=:valid, final_objective=value, final_gradient); point=full)
end

function _run_refiner(r::TNCRefiner, model::QHDProblem, start::AbstractStart)
    artifact = _tnc_installation(r)
    state = try
        ctx = _derivative_context(model; order=1)
        _derivative_point(model, start.point)
        gradient = zeros(length(ctx.free))
        value = objective_value(model, start.point)
        _gradient!(gradient, ctx, start.point)
        _TNCState(ctx, copy(start.point), start.point[ctx.free], value, gradient, 1, 0, nothing)
    catch err
        err isa Union{ArgumentError,DomainError,OverflowError,InexactError} || rethrow()
        return RefinerOutcome(; status=:ineligible_derivatives, solver_called=false, message=sprint(showerror, err))
    end
    free = state.ctx.free
    budget = isnothing(r.maxfun) ? max(100, Base.checked_mul(10, length(free))) : r.maxfun
    max(length(free), budget) <= typemax(Cint) || throw(ArgumentError("problem exceeds TNC's C integer range"))
    scale = isnothing(r.scale) ? C_NULL : Float64[r.scale[i] for i in free]
    offset = isnothing(r.offset) ? C_NULL : Float64[r.offset[i] for i in free]
    if budget > 0
        try
            _tnc_check_scaling(r, state)
        catch err
            err isa ArgumentError || rethrow()
            return RefinerOutcome(; status=:ineligible_scaling, solver_called=false, message=sprint(showerror, err))
        end
    end
    handle = Libdl.dlopen(artifact.path)
    try
        abi = Libdl.dlsym(handle, :qhdopt_tnc_abi)
        ccall(abi, Cint, ()) == 1 || throw(ArgumentError("unsupported TNC C ABI at $(artifact.path)"))
        return _tnc_solve(Libdl.dlsym(handle, :tnc), r, model, start, state.ctx, artifact, budget, scale, offset, state)
    finally
        Libdl.dlclose(handle)
    end
end

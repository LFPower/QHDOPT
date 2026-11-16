module QHDOPTIpoptExt

import QHDOPT
import Ipopt

versions() = (; binding=string(Base.pkgversion(Ipopt)), native=string(Ipopt.GetIpoptVersion()))

# MUMPS thread-safety depends on its build. Keep concurrent callers on the
# serial reference path and keep all Julia callback state local to each call.
const _NATIVE_LOCK = ReentrantLock()

function _point!(full, free, x, model)
    for (k, i) in enumerate(free)
        model.lower[i] <= x[k] <= model.upper[i] ||
            throw(DomainError(x[k], "Ipopt evaluation outside the original box at coordinate $i"))
        full[i] = x[k]
    end
    return full
end

# Ipopt.jl's C wrappers do not relay a Bool from objective/derivative
# callbacks. Return NaN to its invalid-number checks, request termination at
# the next intermediate callback, and report the latched error back in Julia.
function _guard(f, failure, stage, output)
    if isnothing(failure[])
        try
            return f()
        catch err
            failure[] = (stage, err)
        end
    end
    isnothing(output) && return NaN
    fill!(output, NaN)
    return nothing
end

function _native_problem(ctx, start, failure)
    free, model = ctx.free, ctx.model
    full = copy(start.point)
    lower, upper = model.lower[free], model.upper[free]
    # Ipopt treats bounds reaching its infinity sentinels as unbounded.
    if any(==(-floatmax(Float64)), lower) || any(==(floatmax(Float64)), upper)
        throw(ArgumentError("original bound reaches Ipopt's Float64 infinity sentinel"))
    end
    rows, cols = Cint.(ctx.rows), Cint.(ctx.cols)
    n, nnz_h = length(free), length(rows)
    max(n, nnz_h) <= typemax(Cint) || throw(ArgumentError("model exceeds Ipopt's integer index range"))
    function eval_f(x)
        return _guard(failure, :objective, nothing) do
            QHDOPT.objective_value(model, _point!(full, free, x, model))
        end
    end
    function eval_grad(x, values)
        _guard(failure, :gradient, values) do
            QHDOPT._gradient!(values, ctx, _point!(full, free, x, model))
        end
        return
    end
    function eval_h(x, irow, jcol, multiplier, _, values)
        if isnothing(values)
            # The C API supplies no point for this purely structural query.
            # Pattern length, indices, and element type are fixed before C entry.
            try
                copyto!(irow, rows)
                copyto!(jcol, cols)
            catch err
                failure[] = (:hessian_structure, err)
                fill!(irow, Cint(1))
                fill!(jcol, Cint(1))
            end
        else
            _guard(failure, :hessian, values) do
                QHDOPT._hessian_values!(values, ctx, _point!(full, free, x, model), multiplier)
            end
        end
        return
    end
    empty_callback(args...) = nothing  # bound-constrained problems have no constraint/Jacobian entries
    return Ipopt.CreateIpoptProblem(n, lower, upper, 0, Float64[], Float64[], 0, nnz_h,
                                    eval_f, empty_callback, eval_grad, empty_callback, eval_h)
end

function _configure!(prob, refiner, failure)
    Ipopt.AddIpoptStrOption(prob, "option_file_name", "")  # an ambient ipopt.opt must not override these settings
    Ipopt.AddIpoptNumOption(prob, "tol", refiner.tol)
    Ipopt.AddIpoptIntOption(prob, "max_iter", refiner.max_iter)
    Ipopt.AddIpoptStrOption(prob, "hessian_approximation", "exact")
    Ipopt.AddIpoptNumOption(prob, "bound_relax_factor", 0.0)
    Ipopt.AddIpoptNumOption(prob, "nlp_lower_bound_inf", -floatmax(Float64))
    Ipopt.AddIpoptNumOption(prob, "nlp_upper_bound_inf", floatmax(Float64))
    # Receive the caller's exact start. Ipopt then performs its normal,
    # recorded interior-point initialization on variables starting at bounds.
    Ipopt.AddIpoptNumOption(prob, "bound_push", 0.01)
    Ipopt.AddIpoptNumOption(prob, "bound_frac", 0.01)
    Ipopt.AddIpoptStrOption(prob, "honor_original_bounds", "no") # Julia validates the unprojected return
    Ipopt.AddIpoptStrOption(prob, "check_derivatives_for_naninf", "yes")
    Ipopt.AddIpoptStrOption(prob, "linear_solver", "mumps")
    Ipopt.AddIpoptIntOption(prob, "print_level", 0)
    Ipopt.AddIpoptStrOption(prob, "sb", "yes")
    Ipopt.SetIntermediateCallback(prob, (args...) -> isnothing(failure[]))
    return
end

function _termination(code)
    code in (0, 1) && return :converged
    code in (-1, -4, -5) && return :limit_reached
    code == -13 && return :invalid_evaluation
    code == 2 && return :infeasible
    code == 5 && return :stopped
    code == 6 && return :feasible_point
    return :solver_failure
end

function refine(refiner, model, start)
    ctx = try
        context = QHDOPT._derivative_context(model)
        QHDOPT._check_derivative_start(context, start.point)
        context
    catch err
        err isa Union{ArgumentError,DomainError,OverflowError} || rethrow()
        return QHDOPT.RefinerOutcome(; status=:ineligible_derivatives, solver_called=false,
                                     message=sprint(showerror, err))
    end
    return lock(_NATIVE_LOCK) do
        failure = Ref{Any}(nothing)
        prob, called = nothing, false
        try
            prob = _native_problem(ctx, start, failure)
            _configure!(prob, refiner, failure)
            prob.x .= start.point[ctx.free]
            called = true
            code = Int(Ipopt.IpoptSolve(prob))
            if !isnothing(failure[])
                stage, err = failure[]
                err isa InterruptException && throw(err)
                status = err isa Union{ArgumentError,DomainError,OverflowError} ? :invalid_evaluation : :callback_failure
                return QHDOPT.RefinerOutcome(; status, solver_called=true, raw_status=code,
                                             message="$stage callback: $(sprint(showerror, err))")
            end
            # Nonconvergence may still supply a usable final iterate. An
            # initialization or internal error can simply leave prob.x at its
            # input, so never label those buffers as actual solver returns.
            point = if -5 <= code <= 6
                x = copy(start.point)
                x[ctx.free] = prob.x
                x
            else
                nothing
            end
            return QHDOPT.RefinerOutcome(; status=_termination(code), solver_called=true, raw_status=code, point)
        catch err
            err isa InterruptException && rethrow()
            return QHDOPT.RefinerOutcome(; status=:solver_failure, solver_called=called, message=sprint(showerror, err))
        finally
            isnothing(prob) || finalize(prob) # run Ipopt.jl's registered FreeIpoptProblem exactly once
        end
    end
end

end # module

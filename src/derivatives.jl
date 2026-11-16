# Original-coordinate derivatives for gradient and exact-Hessian refiners. Fixed
# variables are substituted, not rescaled. Native row/column indices refer
# to the ordered vector of free coordinates, while evaluators take the full x.
struct DerivativeContext{M,G,H}
    model::M
    free::Vector{Int}
    rows::Vector{Int}
    cols::Vector{Int}
    gradients::G
    hessian::H
end

function _free_positions(model)
    free = findall(model.lower .< model.upper)
    position = zeros(Int, length(model.lower))
    position[free] = eachindex(free)
    return free, position
end

function _derivative_context(model::QHDProblem{QuadraticObjective}; order=2)
    order in (1, 2) || throw(ArgumentError("derivative order must be 1 or 2"))
    free, position = _free_positions(model)
    order == 1 && return DerivativeContext(model, free, Int[], Int[], nothing, nothing)
    entries = Tuple{Int,Int,Float64}[]
    for (i, j, h) in _upper_entries(model.objective.H)
        row, col = position[j], position[i]
        if row > 0 && col > 0
            push!(entries, (row, col, h))
        end
    end
    sort!(entries; by=t -> (t[2], t[1]))
    return DerivativeContext(model, free, Int[t[1] for t in entries], Int[t[2] for t in entries],
                             nothing, Float64[t[3] for t in entries])
end

# Differentiate the original weighted IR, never its encoding tables or a
# separability rewrite. Rules only discard structurally zero derivatives.
function _partial(e::ScalarExpr, i::Int)
    e.op == :literal && return _literal(0)
    e.op == :variable && return _literal(e.coordinate == i ? 1 : 0)
    ds = map(a -> _partial(a, i), e.args)
    all(d -> _is_literal(d, 0), ds) && return _literal(0)
    a, da = first(e.args), first(ds)
    if e.op == :+
        return _sum_expr(ds)
    elseif e.op == :-
        return _operation(:-, ds...)
    elseif e.op == :*
        # Product rule without division by a factor (which might be zero).
        terms = ScalarExpr[]
        for k in eachindex(ds)
            _is_literal(ds[k], 0) && continue
            push!(terms, _product_expr(Base.setindex(e.args, ds[k], k)))
        end
        return _sum_expr(terms)
    elseif e.op == :/
        b, db = last(e.args), last(ds)
        _is_literal(db, 0) && return _operation(:/, da, b)
        numerator = _operation(:-, _operation(:*, da, b), _operation(:*, a, db))
        return _operation(:/, numerator, _power(b, 2.0))
    elseif e.op == :^
        p = e.literal
        iszero(p) && return _literal(0)
        p == 1 && return da
        # The retained IR stores Float64 exponents. For example, with p=0.1
        # the exact dyadic p-1 need not fit. On a zero-free base use a^p/a
        # rather than rounding the derivative's exponent. The box-domain
        # check below rejects the quotient when zero is possible.
        exponent = _exact(p) - 1
        next = Float64(exponent)
        factor = _exact(next) == exponent ? _power(a, next) : _operation(:/, e, a)
        return _operation(:*, _literal(p), factor, da)
    elseif e.op == :exp
        return _operation(:*, e, da)
    elseif e.op == :log
        return _operation(:/, da, a)
    elseif e.op == :sin
        return _operation(:*, _operation(:cos, a), da)
    else  # cos
        return _operation(:-, _operation(:*, _operation(:sin, a), da))
    end
end

function _admit_derivative_box(model, expr, label)
    try
        lo, hi = _with_reference_precision(256) do
            _range(expr, model.lower, model.upper)
        end
        -floatmax(Float64) <= lo <= hi <= floatmax(Float64) ||
            throw(ArgumentError("cannot certify finite Float64 values"))
    catch err
        err isa Union{ArgumentError,DomainError} || rethrow()
        throw(ArgumentError("$label is ineligible on the declared box: $(sprint(showerror, err))"))
    end
    return nothing
end

function _derivative_context(model::QHDProblem{StructuredObjective}; order=2)
    order in (1, 2) || throw(ArgumentError("derivative order must be 1 or 2"))
    free, position = _free_positions(model)
    # The value-domain and the two derivative-domain checks are independent.
    # For example x^1.5 on [0,1] passes the first two, but the represented
    # second derivative contains x^(-0.5) at the included lower endpoint.
    _admit_derivative_box(model, model.objective.root, "objective")
    gradients = [ScalarExpr[] for _ in free]
    hessian = Dict{Tuple{Int,Int},Vector{ScalarExpr}}()
    # Refiners use native exceptions to record derivative ineligibility.
    decomposition = _native_errors(() -> _decompose(model))
    for term in decomposition.terms
        expr = _substitute(term.reference, model)
        # Accepted terms involve at most two free variables. Keep all
        # structurally possible entries, regardless of values at the start.
        for (k, i) in enumerate(term.variables)
            di = _substitute(_partial(expr, i), model)
            _is_literal(di, 0) && continue
            push!(gradients[position[i]], di)
            order == 1 && continue
            for j in @view term.variables[k:end]
                dij = _substitute(_partial(di, j), model)
                _is_literal(dij, 0) && continue
                key = (position[j], position[i])
                push!(get!(() -> ScalarExpr[], hessian, key), dij)
            end
        end
    end
    firsts = [_sum_expr(terms) for terms in gradients]
    indices = sort!(collect(keys(hessian)); by=reverse)
    seconds = [_sum_expr(hessian[key]) for key in indices]
    for (i, expr) in zip(free, firsts)
        _admit_derivative_box(model, expr, "gradient of $(model.names[i])")
    end
    for ((row, col), expr) in zip(indices, seconds)
        i, j = free[row], free[col]
        _admit_derivative_box(model, expr, "Hessian entry ($(model.names[i]), $(model.names[j]))")
    end
    return DerivativeContext(model, free, Int[key[1] for key in indices], Int[key[2] for key in indices],
                             firsts, order == 1 ? nothing : seconds)
end

function _derivative_point(model, x)
    length(x) == length(model.lower) || throw(DimensionMismatch("wrong full objective point dimension"))
    for i in eachindex(x)
        isfinite(x[i]) && model.lower[i] <= x[i] <= model.upper[i] ||
            throw(ArgumentError("coordinate $(model.names[i]) is outside the original finite box"))
    end
end

function _gradient!(dest, ctx::DerivativeContext{<:QHDProblem{QuadraticObjective}}, x)
    _derivative_point(ctx.model, x)
    length(dest) == length(ctx.free) || throw(DimensionMismatch("wrong free-gradient buffer length"))
    # Exact H*x+b retains the contribution of fixed coordinates and avoids
    # losing small gradients to cancellation before the final Float64 round.
    gradient = _reduced_gradient(ctx.model, _exact.(x))
    for (k, i) in enumerate(ctx.free)
        dest[k] = _coefficient(gradient[i], "original gradient at coordinate $i")
    end
    return dest
end

function _gradient!(dest, ctx::DerivativeContext{<:QHDProblem{StructuredObjective}}, x)
    _derivative_point(ctx.model, x)
    length(dest) == length(ctx.free) || throw(DimensionMismatch("wrong free-gradient buffer length"))
    for k in eachindex(dest)
        dest[k] = Float64(_reference(ctx.gradients[k], x).value)
    end
    return dest
end

_hessian_entry(h::Float64, x, multiplier) =
    _coefficient(_exact(h) * _exact(multiplier), "scaled quadratic Hessian")
function _hessian_entry(e::ScalarExpr, x, multiplier)
    scaled = isone(multiplier) ? e : _operation(:*, _literal(multiplier), e)
    return Float64(_reference(scaled, x).value)
end

function _hessian_values!(dest, ctx::DerivativeContext, x, objective_multiplier=1.0)
    isnothing(ctx.hessian) && throw(ArgumentError("this context was prepared for gradients only"))
    _derivative_point(ctx.model, x)
    length(dest) == length(ctx.rows) || throw(DimensionMismatch("wrong Hessian values buffer length"))
    multiplier = _coefficient(objective_multiplier, "Hessian objective multiplier")
    for k in eachindex(dest)
        dest[k] = _hessian_entry(ctx.hessian[k], x, multiplier)
    end
    return dest
end

function _check_derivative_start(ctx::DerivativeContext, x; order=isnothing(ctx.hessian) ? 1 : 2)
    order in (1, 2) || throw(ArgumentError("derivative order must be 1 or 2"))
    _derivative_point(ctx.model, x)
    objective_value(ctx.model, x)
    _gradient!(zeros(length(ctx.free)), ctx, x)
    order == 2 && _hessian_values!(zeros(length(ctx.rows)), ctx, x)
    return nothing
end

const StructuredOperators = QHDOPT._structured_operators

function _literal_value(x::Real)
    x isa Bool && _invalid("Boolean literals are outside the scalar objective grammar")
    value = Float64(x)
    isfinite(value) && value == x || _invalid("objective literal $x is not exactly representable as a finite Float64")
    return value
end

_weighted_term(c, expr) = isone(c) ? expr : QHDOPT._operation(:*, QHDOPT._literal(c), expr)

# Return nothing if storing a derived coefficient would change the polynomial.
function _exact_factor_literal(q)
    v = Float64(q)
    return isfinite(v) && QHDOPT._exact(v) == q ? v : nothing
end

function _affine_factor(slope, i, offset)
    term = _weighted_term(abs(slope), QHDOPT._variable(i))
    slope < 0 && return QHDOPT._operation(:-, QHDOPT._literal(offset), term)
    return iszero(offset) ? term : QHDOPT._operation(:+, term, QHDOPT._literal(offset))
end

function _completed_square(i, h, b, c, lower, upper)
    iszero(b) && return nothing
    # Complete the square only when transformed coefficients are representable.
    shift = b/h
    residual = c - b*shift/2
    remainder = _exact_factor_literal(residual)
    isnothing(remainder) && return nothing
    if !iszero(residual) && signbit(h) != signbit(residual)
        # Avoid introducing a large cancellation between the square and
        # the remainder when the vertex lies far outside this box.
        lo, hi = QHDOPT._exact(lower[i]), QHDOPT._exact(upper[i])
        radius = max(abs(lo), abs(hi))
        shifted_radius = max(abs(lo + shift), abs(hi + shift))
        expanded_scale = abs(c) + abs(b)*radius + abs(h)*radius^2/2
        abs(residual) + abs(h)*shifted_radius^2/2 > expanded_scale && return nothing
    end
    # A non-dyadic vertex can still have a representable affine factor,
    # e.g. (3x-1)², by reducing h*x + b to primitive integer coefficients.
    for scale in (one(h), abs(h)/gcd(h, b))
        slope = _exact_factor_literal(scale)
        offset = _exact_factor_literal(scale*shift)
        weight = _exact_factor_literal(h/(2*scale^2))
        any(isnothing, (slope, offset, weight)) && continue
        square = QHDOPT._power(_affine_factor(slope, i, offset), 2.0)
        term = _weighted_term(weight, square)
        return iszero(remainder) ? term : QHDOPT._operation(:+, QHDOPT._literal(remainder), term)
    end
    return nothing
end

function _bilinear_product(i, j, h, b, d, c, upper)
    iszero(b) && iszero(d) && return nothing
    # Factor only when the exact coefficient identity holds.
    h*c == b*d || return nothing
    for scale in (one(h), abs(h), gcd(h, d))
        a1 = _exact_factor_literal(h/scale)
        b1 = _exact_factor_literal(d/scale)
        a2 = _exact_factor_literal(scale)
        b2 = _exact_factor_literal(scale*b/h)
        any(isnothing, (a1, b1, a2, b2)) && continue
        # Flip both factors when the second is nonpositive throughout the box.
        # A positive product can then pass the native log-product domain check.
        sign = QHDOPT._exact(upper[j]) + b/h <= 0 ? -1 : 1
        left = _affine_factor(sign*a1, i, sign*b1)
        right = _affine_factor(sign*a2, j, sign*b2)
        return QHDOPT._operation(:*, left, right)
    end
    return nothing
end

# JuMP expands squares and products of affine factors before passing them
# to MOI. Restore these forms only when the exact coefficient identities hold.
function _factored_quadratic(i, j, h, sums, constant, lower, upper, checkpoint)
    for (k, (v, b)) in enumerate(sums)
        _translation_checkpoint(checkpoint, k)
        v != i && v != j && !iszero(b) && return nothing
    end
    b = get(sums, i, 0)
    _checked_sum(b)
    c = QHDOPT._exact(constant)
    i == j && return _completed_square(i, h, b, c, lower, upper)
    d = get(sums, j, 0)
    _checked_sum(d)
    return _bilinear_product(i, j, h, b, d, c, upper)
end

# JuMP keeps affine and quadratic subexpressions as MOI functions inside
# nonlinear nodes. Coalesce their raw terms using the same policy as box QPs.
function _polynomial_expr(f, indices, lower, upper, checkpoint)
    terms = QHDOPT.ScalarExpr[]
    constant = _literal_value(f.constant)
    iszero(constant) || push!(terms, QHDOPT._literal(constant))
    affine = f isa MOI.ScalarQuadraticFunction ? f.affine_terms : f.terms
    sums = _affine_sums(affine, indices, checkpoint)
    if f isa MOI.ScalarQuadraticFunction
        entries = _quadratic_sums(f.quadratic_terms, indices, checkpoint)
        nconstant = length(terms)
        last_entry = nothing
        for (k, ((i, j), total)) in enumerate(entries)
            _translation_checkpoint(checkpoint, k)
            iszero(total) && continue
            c = _checked_sum(total)
            x = QHDOPT._variable(i)
            if i == j
                # Halving c directly could underflow before multiplication by x^2.
                term = QHDOPT._operation(:*, QHDOPT._literal(0.5), QHDOPT._power(x, 2.0))
            else
                term = QHDOPT._operation(:*, x, QHDOPT._variable(j))
            end
            push!(terms, _weighted_term(c, term))
            last_entry = (i, j, total)
        end
        if length(terms) == nconstant + 1
            factored = _factored_quadratic(last_entry..., sums, constant, lower, upper, checkpoint)
            !isnothing(factored) && return factored
        end
    end
    for (k, (i, total)) in enumerate(sums)
        _translation_checkpoint(checkpoint, k)
        iszero(total) || push!(terms, _weighted_term(_checked_sum(total), QHDOPT._variable(i)))
    end
    return QHDOPT._sum_expr(terms)
end

# Bound recursive descent at ingestion. The native compiler also recurses on
# this tree; polynomial leaves add only fixed-depth arithmetic, and Max adds
# one negation. Wide sums and products do not consume additional depth.
const _MAX_NONLINEAR_DEPTH = 128

function _structured_expr(node, indices, lower, upper, checkpoint, ancestors)::QHDOPT.ScalarExpr
    checkpoint()
    if node isa Real
        return QHDOPT._literal(_literal_value(node))
    elseif node isa MOI.VariableIndex
        return QHDOPT._variable(_objective_coordinate(indices, node))
    elseif node isa Union{MOI.ScalarAffineFunction{Float64},MOI.ScalarQuadraticFunction{Float64}}
        return _polynomial_expr(node, indices, lower, upper, checkpoint)
    end
    node isa MOI.ScalarNonlinearFunction || _invalid("unsupported nonlinear objective argument of type $(typeof(node))")
    op, args = node.head, node.args
    op in StructuredOperators || throw(MOI.UnsupportedNonlinearOperator(op))
    args in ancestors && _invalid("cyclic nonlinear objective")
    length(ancestors) < _MAX_NONLINEAR_DEPTH ||
        _invalid("nonlinear objective exceeds $_MAX_NONLINEAR_DEPTH nested operators; simplify the expression")
    if isempty(args) && op in (:+, :*)
        return QHDOPT._literal(op == :+ ? 0.0 : 1.0)
    end
    n = length(args)
    QHDOPT._valid_scalar_arity(op, n) || _invalid("nonlinear operator $op does not accept $n arguments")
    push!(ancestors, args)
    if op == :^
        args[2] isa Real || _invalid("power requires a finite numeric literal exponent")
        base = _structured_expr(args[1], indices, lower, upper, checkpoint, ancestors)
        expr = QHDOPT._power(base, _literal_value(args[2]))
    else
        children = [_structured_expr(arg, indices, lower, upper, checkpoint, ancestors) for arg in args]
        expr = QHDOPT._operation(op, children...)
    end
    delete!(ancestors, args)
    return expr
end

function _structured_model(f, indices, sense, lower, upper, checkpoint)
    root = _structured_expr(f, indices, lower, upper, checkpoint, Base.IdSet{Vector{Any}}())
    sense == MOI.MAX_SENSE && (root = QHDOPT._operation(:-, root))
    checkpoint()
    objective = QHDOPT.StructuredObjective(root, length(lower))
    try
        return QHDOPT.QHDProblem(objective, lower, upper, nothing, checkpoint)
    catch err
        err isa ArgumentError || rethrow()
        _invalid(sprint(showerror, err))
    end
end

# Package-owned scalar syntax. Children are immutable, so neither a caller's
# Expr nor a later edit to its argument vector can change an objective.
struct ScalarExpr
    op::Symbol
    literal::Float64
    coordinate::Int
    args::Tuple{Vararg{ScalarExpr}}
end

struct StructuredObjective
    root::ScalarExpr
    dimension::Int
end

struct StructuredTerm
    reference::ScalarExpr       # full weighted original expression
    weight::Float64
    kind::Symbol
    variables::Vector{Int}      # original indices, sorted
    factors::Vector{ScalarExpr}
end

_literal(x) = ScalarExpr(:literal, _coefficient(x, "structured literal"), 0, ())
_variable(i) = ScalarExpr(:variable, 0.0, i, ())
_operation(op, args...) = ScalarExpr(op, 0.0, 0, args)
_power(arg, p) = ScalarExpr(:^, p, 0, (arg,))

_negates(a, b, same) = a.op == :- && length(a.args) == 1 && same(only(a.args), b)
_opposite_pair(args, same=(==)) =
    length(args) == 2 && (_negates(args[1], args[2], same) || _negates(args[2], args[1], same))

# Restrict fractional exponent literals to exact Float64 values so that a
# rounded integer cannot accidentally broaden the admitted real domain.
function _exponent_literal(x)
    value = if x isa Real && !(x isa Bool)
        isfinite(x) || throw(ArgumentError("power needs a finite literal exponent"))
        _exact(x)
    elseif x isa Expr && x.head == :call && x.args[1] in (:/, ://) && length(x.args) == 3
        a, b = _exponent_literal(x.args[2]), _exponent_literal(x.args[3])
        iszero(b) && throw(ArgumentError("zero denominator in exponent"))
        _exact(a) / _exact(b)
    elseif x isa Expr && x.head == :call && x.args[1] == :- && length(x.args) == 2
        -_exact(_exponent_literal(x.args[2]))
    else
        throw(ArgumentError("power needs a numeric literal exponent, got $x"))
    end
    p = _coefficient(value, "exponent")
    _exact(p) == value || throw(ArgumentError("exponent $x is not exactly representable as Float64"))
    return p
end

const _structured_univariate = (:sin, :cos, :exp, :log)
const _structured_operators = (:+, :-, :*, :/, :^, _structured_univariate...)

function _valid_scalar_arity(op, n)
    return (op in (:+, :*) && n >= 1) || (op == :- && n in (1, 2)) ||
           (op in (:/, :^) && n == 2) || (op in _structured_univariate && n == 1)
end

function _parse_scalar(x, names)
    if x isa Real && !(x isa Bool)
        return _literal(x)
    elseif x isa Symbol
        i = findfirst(==(x), names)
        isnothing(i) && throw(ArgumentError("undeclared variable $x"))
        return _variable(i)
    end
    x isa Expr && x.head == :call && !isempty(x.args) ||
        throw(ArgumentError("expected a quoted scalar expression, got $(repr(x))"))
    op, args = first(x.args), x.args[2:end]
    _valid_scalar_arity(op, length(args)) || throw(ArgumentError("unsupported scalar operation or arity in $(repr(x))"))
    if op == :^
        return _power(_parse_scalar(args[1], names), _exponent_literal(args[2]))
    end
    return _operation(op, (_parse_scalar(a, names) for a in args)...)
end

function _expr_text(e, names=nothing, checkpoint=Returns(nothing))
    checkpoint()
    e.op == :literal && return string(e.literal)
    e.op == :variable && return isnothing(names) ? "x[$(e.coordinate)]" : string(names[e.coordinate])
    args = [_expr_text(child, names, checkpoint) for child in e.args]
    e.op == :^ && return "($(only(args)))^$(e.literal)"
    e.op in (:+, :-, :*, :/) && length(args) > 1 && return "(" * join(args, " $(e.op) ") * ")"
    return "$(e.op)(" * join(args, ", ") * ")"
end

function _evaluate(e::ScalarExpr, x, checkpoint=Returns(nothing))
    checkpoint()
    T = eltype(x)
    e.op == :literal && return T(e.literal)
    e.op == :variable && return x[e.coordinate]
    args = map(a -> _evaluate(a, x, checkpoint), e.args)
    value = if e.op == :+
        sum(args)
    elseif e.op == :*
        prod(args)
    elseif e.op == :-
        -(args...)
    elseif e.op == :/
        args[1]/args[2]
    elseif e.op == :^
        only(args)^T(e.literal)
    elseif e.op == :exp
        exp(only(args))
    elseif e.op == :log
        log(only(args))
    elseif e.op == :sin
        sin(only(args))
    else
        cos(only(args))
    end
    isfinite(value) && !(e.op == :exp && iszero(value)) ||
        throw(ArgumentError("nonfinite value or lost positive exponential at $(_expr_text(e, nothing, checkpoint))"))
    return value
end

# Outward-rounded interval operations are used both to prove real domains
# before simplification and to resolve the *original*, unseparated value.
_down(f) = setrounding(f, BigFloat, RoundDown)
_up(f) = setrounding(f, BigFloat, RoundUp)

# BigFloat contexts are task-local in Julia 1.13. Serialize package-owned
# context changes on older supported Julia versions too.
const _REFERENCE_LOCK = ReentrantLock()
_with_reference_precision(f, bits) = lock(_REFERENCE_LOCK) do
    setprecision(f, BigFloat, bits)
end

function _interval_product(a, b)
    # Zero times an unbounded range remains zero when one factor is zero.
    mul(x, y) = iszero(x) || iszero(y) ? zero(BigFloat) : x*y
    lo = _down(() -> minimum(mul(x, y) for x in a for y in b))
    hi = _up(() -> maximum(mul(x, y) for x in a for y in b))
    return lo, hi
end

function _endpoint_sum(endpoint, intervals, checkpoint)
    return sum(intervals) do interval
        checkpoint()
        endpoint(interval)
    end
end

function _power_range(range, p, expr, checkpoint)
    lo, hi = range
    !isinteger(p) && lo < 0 && throw(ArgumentError("cannot prove a nonnegative base for $(_expr_text(expr, nothing, checkpoint))"))
    p < 0 && lo <= 0 <= hi && throw(ArgumentError("negative power at possible zero in $(_expr_text(expr, nothing, checkpoint))"))
    iszero(p) && return (one(BigFloat), one(BigFloat))
    lower = _down(() -> min(lo^p, hi^p))
    upper = _up(() -> max(lo^p, hi^p))
    # A positive even integral power can attain zero inside the interval.
    if p > 0 && lo <= 0 <= hi && isinteger(p) && iseven(BigInt(p))
        lower = zero(BigFloat)
    end
    return lower, upper
end

function _trig_range(op, range)
    lo, hi = range
    f = op == :sin ? sin : cos
    isfinite(lo) && isfinite(hi) || return (-one(BigFloat), one(BigFloat))
    # Both functions are 1-Lipschitz. Centering at one endpoint also encloses
    # narrow arguments returned by a previous inexact operation at a point.
    width = _up(() -> hi-lo)
    lower = _down(() -> f(lo) - width)
    upper = _up(() -> f(lo) + width)
    return max(-one(BigFloat), lower), min(one(BigFloat), upper)
end

# A repeated subexpression at an exact point may have a nonzero enclosure
# width by itself. Recognize equality before subtracting its two enclosures
# (e.g. sin(x)-sin(1) at x=1). Only equal fixed values or identical syntax
# can establish this relation.
function _equal_on_box(a, b, lower, upper, checkpoint)
    checkpoint()
    if a.op in (:variable, :literal) && b.op in (:variable, :literal)
        a.op == b.op == :variable && a.coordinate == b.coordinate && return true
        a.op == :variable && lower[a.coordinate] != upper[a.coordinate] && return false
        b.op == :variable && lower[b.coordinate] != upper[b.coordinate] && return false
        av = a.op == :literal ? a.literal : lower[a.coordinate]
        bv = b.op == :literal ? b.literal : lower[b.coordinate]
        return av == bv
    end
    a.op == b.op && a.literal == b.literal && length(a.args) == length(b.args) || return false
    return all(_equal_on_box(x, y, lower, upper, checkpoint) for (x, y) in zip(a.args, b.args))
end

function _range(e, lower, upper, checkpoint=Returns(nothing))
    checkpoint()
    e.op == :literal && return (BigFloat(e.literal), BigFloat(e.literal))
    e.op == :variable && return (BigFloat(lower[e.coordinate], RoundDown),
                                 BigFloat(upper[e.coordinate], RoundUp))
    args = map(child -> _range(child, lower, upper, checkpoint), e.args)
    a = first(args)
    result = if e.op == :+
        if _opposite_pair(e.args, (x,y) -> _equal_on_box(x,y,lower,upper,checkpoint))
            (zero(BigFloat), zero(BigFloat))
        else
            (_down(() -> _endpoint_sum(first, args, checkpoint)),
             _up(() -> _endpoint_sum(last, args, checkpoint)))
        end
    elseif e.op == :-
        if length(args) == 1
            (-a[2], -a[1])
        elseif _equal_on_box(e.args[1], e.args[2], lower, upper, checkpoint)
            # The arguments have already passed domain checks.
            (zero(BigFloat), zero(BigFloat))
        else
            b = args[2]
            (_down(() -> a[1]-b[2]), _up(() -> a[2]-b[1]))
        end
    elseif e.op == :*
        reduce(args) do left, right
            checkpoint()
            _interval_product(left, right)
        end
    elseif e.op == :/
        b = args[2]
        b[1] <= 0 <= b[2] && throw(ArgumentError("possible zero denominator in $(_expr_text(e, nothing, checkpoint))"))
        if _equal_on_box(e.args[1], e.args[2], lower, upper, checkpoint)
            (one(BigFloat), one(BigFloat))
        else
            _interval_product(a, (_down(() -> inv(b[2])), _up(() -> inv(b[1]))))
        end
    elseif e.op == :^
        _power_range(a, BigFloat(e.literal), e, checkpoint)
    elseif e.op == :exp
        (_down(() -> exp(a[1])), _up(() -> exp(a[2])))
    elseif e.op == :log
        a[1] > 0 || throw(ArgumentError("cannot prove positive logarithm argument in $(_expr_text(e, nothing, checkpoint))"))
        (_down(() -> log(a[1])), _up(() -> log(a[2])))
    else
        _trig_range(e.op, a)
    end
    any(isnan, result) && throw(ArgumentError("unresolved interval for $(_expr_text(e, nothing, checkpoint))"))
    return result
end

function _reference(e::ScalarExpr, x, checkpoint=Returns(nothing))
    for bits in (256, 512, 1024, 2048, 4096)
        attempted = _with_reference_precision(bits) do
            lo, hi = _range(e, x, x, checkpoint)
            l64, h64 = Float64(lo), Float64(hi)
            # Require a uniquely determined finite Float64; only a proven zero
            # may round to zero. The interval is also retained for error bounds.
            isfinite(l64) && l64 == h64 && (!iszero(l64) || iszero(lo) && iszero(hi)) ||
                return nothing
            return (value=lo + (hi-lo)/2, lower=lo, upper=hi, precision=bits)
        end
        isnothing(attempted) || return attempted
    end
    throw(ArgumentError("unresolved, underflowed, or nonfinite original value of $(_expr_text(e, nothing, checkpoint)) at $x"))
end

# Reference evaluation during compilation runs only after domain admission.
# Preserve the original exception for ordinary objective/derivative callers.
function _compiled_reference(e, x, checkpoint=Returns(nothing))
    try
        return _reference(e, x, checkpoint)
    catch err
        err isa Union{ArgumentError,DomainError,OverflowError,InexactError} || rethrow()
        _numerical_admission(err)
    end
end

function objective_value(objective::StructuredObjective, x::AbstractVector{<:Real})
    length(x) == objective.dimension || throw(DimensionMismatch("wrong structured point dimension"))
    all(isfinite, x) || throw(ArgumentError("structured objective point must be finite"))
    return Float64(_reference(objective.root, x).value)
end

"""
    structured_problem(expr; names, lower=0, upper=1)

Parse an owned, original-coordinate scalar expression, e.g.
`structured_problem(:(x^2 + x*y*exp(x)); names=[:x,:y])`. Symbols must be
declared in `names`. The scalar grammar supports finite real literals, +, -,
*, /, constant powers, sin, cos, log, and exp. No Julia code is evaluated.
Compile-time decomposition accepts univariate terms and separable pairs.
"""
function structured_problem(expr; names, lower=0, upper=1)
    names isa AbstractVector && !isempty(names) || throw(ArgumentError("supply ordered variable names"))
    L, U, labels = _model_box(length(names), lower, upper, names)
    root = _parse_scalar(expr, labels)
    return QHDProblem(StructuredObjective(root, length(labels)), L, U, labels)
end

function QHDProblem(objective::StructuredObjective, lower, upper, names, checkpoint=Returns(nothing))
    L, U, labels = _model_box(objective.dimension, lower, upper, names)
    try
        _with_reference_precision(256) do
            _range(objective.root, L, U, checkpoint)
        end
    catch err
        err isa ArgumentError || rethrow()
        throw(ArgumentError("domain of $(_expr_text(objective.root, labels, checkpoint)): $(err.msg)"))
    end
    return QHDProblem(Val(:owned), objective, L, U, labels)
end

_snapshot(model::QHDProblem{StructuredObjective}, checkpoint=Returns(nothing)) =
    QHDProblem(model.objective, model.lower, model.upper, model.names, checkpoint)

_is_literal(e, x) = e.op == :literal && e.literal == x
_sum_expr(args) = isempty(args) ? _literal(0) : length(args) == 1 ? only(args) : _operation(:+, args...)
_product_expr(args) = isempty(args) ? _literal(1) : length(args) == 1 ? only(args) : _operation(:*, args...)
_signed(e, sign) = sign == 1 ? e : _operation(:-, e)

# Apply exact structural identities only after the entire original box has
# passed domain checks. Leave nontrivial constant arithmetic in expression
# form so separate Float64 folds cannot hide an underflow or cancellation.
function _substitute(e, model, checkpoint=Returns(nothing))
    checkpoint()
    if e.op == :variable
        i = e.coordinate
        return model.lower[i] == model.upper[i] ? _literal(model.lower[i]) : e
    end
    e.op == :literal && return e
    args = map(child -> _substitute(child, model, checkpoint), e.args)
    if e.op == :*
        any(a -> _is_literal(a, 0), args) && return _literal(0)
        for a in args
            # Constant transcendental factors such as sin(0) can also be
            # proven zero. Evaluate only their range; a large *nonzero*
            # multiplier need not itself fit Float64 at this stage.
            if a.op != :literal && isempty(_coordinates(a, checkpoint))
                lo, hi = _with_reference_precision(256) do
                    _range(a, model.lower, model.lower, checkpoint)
                end
                iszero(lo) && iszero(hi) && return _literal(0)
            end
        end
        return _product_expr(filter(a -> !_is_literal(a, 1), args))
    elseif e.op == :+
        _opposite_pair(args) && return _literal(0)
        return _sum_expr(filter(a -> !_is_literal(a, 0), args))
    elseif e.op == :- && length(args) == 2 && args[1] == args[2]
        return _literal(0)
    elseif e.op == :/ && _is_literal(first(args), 0)
        return _literal(0)
    elseif e.op == :/ && args[1] == args[2]
        return _literal(1)
    elseif e.op == :^ && e.literal in (0, 1)
        return iszero(e.literal) ? _literal(1) : only(args)
    end
    return ScalarExpr(e.op, e.literal, e.coordinate, args)
end

function _coordinates(e, checkpoint=Returns(nothing))
    indices = Int[]
    function visit(node)
        checkpoint()
        node.op == :variable && push!(indices, node.coordinate)
        foreach(visit, node.args)
    end
    visit(e)
    return sort!(unique!(indices))
end

function _sum_parts!(out, e, sign, checkpoint)
    checkpoint()
    if e.op == :+
        foreach(a -> _sum_parts!(out, a, sign, checkpoint), e.args)
    elseif e.op == :-
        length(e.args) == 2 && _sum_parts!(out, first(e.args), sign, checkpoint)
        _sum_parts!(out, last(e.args), -sign, checkpoint)
    else
        push!(out, _signed(e, sign))
    end
    return out
end

function _product_parts!(out, e, checkpoint)
    checkpoint()
    if e.op == :*
        foreach(a -> _product_parts!(out, a, checkpoint), e.args)
    elseif e.op == :- && length(e.args) == 1
        push!(out, _literal(-1))
        _product_parts!(out, only(e.args), checkpoint)
    elseif e.op == :/
        _product_parts!(out, first(e.args), checkpoint)
        push!(out, _operation(:/, _literal(1), last(e.args)))
    else
        push!(out, e)
    end
    return out
end

_unsupported_decomposition(message) = throw(_SolveFailure(:invalid_model, :decomposition, ArgumentError(message)))

# Group factors (or exponent addends) in original variable order. Put any
# inner constant in the first group, so e.g. exp(x+y+7) retains all of 7.
function _group_variables(parts, variables, joiner, label, checkpoint)
    groups = [ScalarExpr[] for _ in variables]
    for part in parts
        coords = _coordinates(part, checkpoint)
        length(coords) <= 1 || _unsupported_decomposition("nonseparable factor in $label")
        index = isempty(coords) ? 1 : findfirst(==(only(coords)), variables)
        push!(groups[index], part)
    end
    return [joiner(group) for group in groups]
end

function _structured_term(reference, reduced, model, checkpoint)
    label = _expr_text(reference, model.names, checkpoint)
    variables = _coordinates(reduced, checkpoint)
    length(variables) <= 2 ||
        _unsupported_decomposition("$label spans more than two free variables $(model.names[variables])")
    parts = _product_parts!(ScalarExpr[], reduced, checkpoint)
    constants = ScalarExpr[]
    factors = ScalarExpr[]
    for part in parts
        push!(isempty(_coordinates(part, checkpoint)) ? constants : factors, part)
    end
    weight = Float64(_compiled_reference(_product_expr(constants), model.lower, checkpoint).value)
    if length(variables) == 1
        return StructuredTerm(reference, weight, :univariate, variables, [_product_expr(factors)])
    end
    if length(factors) == 1
        body = only(factors)
        if body.op == :exp
            pieces = _sum_parts!(ScalarExpr[], only(body.args), 1, checkpoint)
            arguments = _group_variables(pieces, variables, _sum_expr, label, checkpoint)
            return StructuredTerm(reference, weight, :exp_sum, variables, arguments)
        elseif body.op == :log || body.op == :^
            pieces = _product_parts!(ScalarExpr[], only(body.args), checkpoint)
            arguments = _group_variables(pieces, variables, _product_expr, label, checkpoint)
            for arg in arguments
                lower, _ = _with_reference_precision(256) do
                    _range(arg, model.lower, model.upper, checkpoint)
                end
                body.op == :log && lower <= 0 &&
                    _unsupported_decomposition("logarithm rewrite of $label needs strictly positive factors")
                body.op == :^ && !isinteger(body.literal) && lower < 0 &&
                    _unsupported_decomposition("power rewrite of $label needs nonnegative factors")
            end
            if body.op == :log
                return StructuredTerm(reference, weight, :log_product, variables,
                                      [_operation(:log, arg) for arg in arguments])
            end
            return StructuredTerm(reference, weight, :product, variables,
                                  [_power(arg, body.literal) for arg in arguments])
        end
    end
    grouped = _group_variables(factors, variables, _product_expr, label, checkpoint)
    return StructuredTerm(reference, weight, :product, variables, grouped)
end

function _decompose(model::QHDProblem{StructuredObjective}, checkpoint=Returns(nothing))
    terms, constants = StructuredTerm[], ScalarExpr[]
    function collect_term(original, sign=1, multipliers=())
        reference = _signed(_product_expr((multipliers..., original)), sign)
        reduced = _substitute(reference, model, checkpoint)
        if isempty(_coordinates(reduced, checkpoint))
            push!(constants, reference)
        elseif original.op == :+
            foreach(arg -> collect_term(arg, sign, multipliers), original.args)
        elseif original.op == :-
            length(original.args) == 2 && collect_term(first(original.args), sign, multipliers)
            collect_term(last(original.args), -sign, multipliers)
        else
            if original.op == :*
                # Distribute a scalar, including fixed-only factors, over a
                # sum. Preserve the full multiplier in each local reference.
                parts = _product_parts!(ScalarExpr[], original, checkpoint)
                active = findall(a -> !isempty(_coordinates(_substitute(a, model, checkpoint), checkpoint)), parts)
                if length(active) == 1 && parts[only(active)].op in (:+, :-)
                    i = only(active)
                    scalar = Tuple(a for (j, a) in enumerate(parts) if j != i)
                    collect_term(parts[i], sign, (multipliers..., scalar...))
                    return
                end
            end
            push!(terms, _structured_term(reference, reduced, model, checkpoint))
        end
    end
    collect_term(model.objective.root)
    reference = _compiled_reference(_sum_expr(constants), model.lower, checkpoint)
    constant = Float64(reference.value)
    constant_error = _with_reference_precision(reference.precision) do
        err = _up(() -> max(BigFloat(constant) - reference.lower, reference.upper - BigFloat(constant)))
        Float64(err, RoundUp)
    end
    return (; constant, constant_error, terms)
end

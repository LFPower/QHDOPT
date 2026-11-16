"""A quadratic objective in the coordinates of its model."""
struct QuadraticObjective
    H::CoefficientMatrix
    linear::Vector{Float64}
    constant::Float64
end

struct QHDProblem{O}
    objective::O
    lower::Vector{Float64}
    upper::Vector{Float64}
    names::Vector{Symbol}
    # The public constructors both pass through input validation and copying.
    QHDProblem(::Val{:owned}, objective::O, lower, upper, names) where {O} =
        new{O}(objective, lower, upper, names)
end

_exact(x::Real) = Rational{BigInt}(x)

# The optimizer needs the reason for a failure, while existing native callers
# depend on the original exception type. Only known checks tag their errors.
struct _SolveFailure <: Exception
    kind::Symbol
    stage::Symbol
    cause::Exception
end

function Base.showerror(io::IO, err::_SolveFailure)
    print(io, err.stage, ": ")
    showerror(io, err.cause)
end

function _native_errors(f)
    try
        return f()
    catch err
        err isa _SolveFailure || rethrow()
        rethrow(err.cause)
    end
end

_numerical_admission(err) = throw(_SolveFailure(:numerical_admission, :compilation, err))
_invalid_option(stage, err) = throw(_SolveFailure(:invalid_option, stage, err))

function _coefficient(x, label, on_error=throw)
    value = Float64(x)
    isfinite(value) && (!iszero(value) || iszero(x)) ||
        on_error(ArgumentError("$label is not representable as a finite Float64 coefficient"))
    return value
end

_compiled_coefficient(x, label) = _coefficient(x, label, _numerical_admission)

function _bound_vector(bounds, n, name)
    values = bounds isa Real ? fill(bounds, n) : bounds
    values isa AbstractVector{<:Real} || throw(ArgumentError("$name must be real bounds"))
    length(values) == n || throw(DimensionMismatch("$name must have $n entries"))
    return values
end

function _model_box(n, lower, upper, names)
    low = _bound_vector(lower, n, "lower")
    high = _bound_vector(upper, n, "upper")
    L, U = _vector(low, "lower"), _vector(high, "upper")
    for i in 1:n
        _coefficient(low[i], "lower[$i]")
        _coefficient(high[i], "upper[$i]")
        low[i] <= high[i] || throw(ArgumentError("reversed bounds at coordinate $i"))
        low[i] < high[i] && L[i] == U[i] &&
            throw(ArgumentError("bounds at coordinate $i collapse in Float64"))
    end
    labels = if isnothing(names)
        [Symbol("x", i) for i in 1:n]
    else
        names isa AbstractVector && length(names) == n ||
            throw(DimensionMismatch("provide one name per coordinate"))
        all(name -> name isa Union{Symbol,AbstractString}, names) ||
            throw(ArgumentError("names must be symbols or strings"))
        Symbol.(collect(names))
    end
    allunique(labels) || throw(ArgumentError("variable names must be unique"))
    return L, U, labels
end

"""
    quadratic_problem(H, b; constant=0, lower=0, upper=1, names=nothing)

Construct an owned original-coordinate model for `constant + x'H*x/2 + b'x`.
Supply a scalar bound for every variable or one entry per coordinate. The
Hessian must be exactly symmetric. Dense strided and sparse CSC inputs are
supported. Equal bounds fix a coordinate.
"""
function quadratic_problem(H, b; constant::Real=0, lower=0, upper=1, names=nothing)
    quadratic, linear = _matrix(H, "H"), _vector(b, "b")
    n = length(linear)
    n > 0 && size(quadratic) == (n, n) ||
        throw(DimensionMismatch("H must be nonempty, square, and match b"))
    issymmetric(H) || throw(ArgumentError("H must be exactly symmetric"))
    # Distinguish true input zeros/fixed bounds from roundoff during admission.
    original_entries = H isa SparseMatrixCSC ? nonzeros(H) : H
    for (i, h) in enumerate(original_entries)
        iszero(h) || !iszero(Float64(h)) || throw(ArgumentError("H entry $i underflows Float64"))
    end
    for i in eachindex(linear)
        _coefficient(b[i], "b[$i]")
    end
    L, U, labels = _model_box(n, lower, upper, names)
    return QHDProblem(Val(:owned), QuadraticObjective(quadratic, linear, _coefficient(constant, "constant")),
                      L, U, labels)
end

QHDProblem(objective::QuadraticObjective, lower, upper, names) =
    quadratic_problem(objective.H, objective.linear; constant=objective.constant, lower, upper, names)

_snapshot(model::QHDProblem{QuadraticObjective}) =
    QHDProblem(model.objective, model.lower, model.upper, model.names)

# Each distinct Hessian entry contributes once, in column order in both
# storage modes. A sparse model never requires a dense scan.
_upper_entries(H::Matrix) =
    ((i, j, H[i,j]) for j in axes(H, 2) for i in 1:j if !iszero(H[i,j]))
_upper_entries(H::SparseMatrixCSC) =
    ((H.rowval[k], j, H.nzval[k]) for j in axes(H, 2) for k in H.colptr[j]:H.colptr[j+1]-1
     if H.rowval[k] <= j && !iszero(H.nzval[k]))

function _exact_value(objective, x)
    value = _exact(objective.constant)
    for i in eachindex(x)
        value += _exact(objective.linear[i]) * x[i]
    end
    for (i, j, h) in _upper_entries(objective.H)
        term = _exact(h) * x[i] * x[j]
        value += i == j ? term / 2 : term
    end
    return value
end

"""Evaluate the original quadratic, accumulating exactly before rounding to Float64."""
function objective_value(objective::QuadraticObjective, x::AbstractVector{<:Real})
    length(x) == length(objective.linear) || throw(DimensionMismatch("wrong point dimension"))
    all(isfinite, x) || throw(ArgumentError("objective point must be finite"))
    return _coefficient(_exact_value(objective, _exact.(x)), "objective at the requested point")
end

objective_value(model::QHDProblem, x) = objective_value(model.objective, x)

struct BoxTransform
    free::Vector{Int}
    fixed::Vector{Int}
    lower::Vector{Float64}
    upper::Vector{Float64}
    widths::Union{Nothing,Vector{Float64}}  # absent only on a shortcut with an unrepresentable width
end

function BoxTransform(model::QHDProblem)
    free = findall(model.upper .> model.lower)
    fixed = findall(model.upper .== model.lower)
    widths = model.upper[free] - model.lower[free]
    return BoxTransform(free, fixed, copy(model.lower), copy(model.upper),
                        all(isfinite, widths) ? widths : nothing)
end

"""Insert fixed coordinates and map the free unit-box coordinates to the original box."""
function reconstruct(transform::BoxTransform, z)
    length(z) == length(transform.free) || throw(DimensionMismatch("wrong free-coordinate count"))
    x = copy(transform.lower)
    for (j, i) in enumerate(transform.free)
        t = z[j]
        t isa Real && 0 <= t <= 1 || throw(ArgumentError("invalid unit coordinate $j"))
        # Exact interpolation avoids cancellation at a narrow or distant endpoint.
        x[i] = Float64((1 - _exact(t)) * _exact(transform.lower[i]) + _exact(t) * _exact(transform.upper[i]))
    end
    return x
end

function _reduced_gradient(model, anchor)
    gradient = _exact.(model.objective.linear)
    for (i, j, h) in _upper_entries(model.objective.H)
        gradient[i] += _exact(h) * anchor[j]
        i != j && (gradient[j] += _exact(h) * anchor[i])
    end
    return gradient
end

# Determine constancy on the *continuous* reduced box using the input
# Float64 values as exact dyadics. An underflowed linear term or a grid-flat
# nonconstant polynomial must never qualify for a deterministic shortcut.
function _normalize(model, transform)
    anchor = _exact.(model.lower)
    free = transform.free
    gradient = _reduced_gradient(model, anchor)[free]
    H = model.objective.H[free, free]
    anchor_value = _exact_value(model.objective, anchor)
    constant = _compiled_coefficient(anchor_value, "objective at lower bounds")
    shortcut = if isempty(free)
        :all_fixed
    elseif !_has_nonzero(H) && all(iszero, gradient)
        :constant
    else
        :none
    end
    if shortcut != :none
        return QuadraticObjective(H, zeros(length(free)), constant), shortcut, gradient
    end
    isnothing(transform.widths) && _numerical_admission(ArgumentError("nonfinite free-coordinate width"))
    widths = _exact.(model.upper[free]) - anchor[free]
    # A conservative bound admits finite values throughout the box without
    # enumerating the Cartesian grid. Use exact coefficients before rounding.
    exact_linear = widths .* gradient
    magnitude_bound = abs(anchor_value) + sum(abs, exact_linear)
    for (i, j, h) in _upper_entries(H)
        coefficient = widths[i] * _exact(h) * widths[j]
        magnitude_bound += abs(coefficient) / (i == j ? 2 : 1)
        v = _compiled_coefficient(coefficient, "normalized H[$i,$j]")
        H[i,j] = H[j,i] = v
    end
    magnitude_bound <= _exact(floatmax(Float64)) ||
        _numerical_admission(ArgumentError("cannot certify finite Float64 objective values across the box"))
    linear = [_compiled_coefficient(exact_linear[i], "normalized linear[$i]") for i in eachindex(free)]
    return QuadraticObjective(H, linear, constant), shortcut, exact_linear
end

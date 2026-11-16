const BoundSet = Union{
    MOI.LessThan{Float64}, MOI.GreaterThan{Float64}, MOI.EqualTo{Float64}, MOI.Interval{Float64},
}
const PolynomialObjective = Union{
    MOI.VariableIndex, MOI.ScalarAffineFunction{Float64}, MOI.ScalarQuadraticFunction{Float64},
}
const ObjectiveFunction = Union{PolynomialObjective,MOI.ScalarNonlinearFunction}

struct TranslationFailure <: Exception
    status::MOI.TerminationStatusCode
    message::String
end
Base.showerror(io::IO, err::TranslationFailure) = print(io, err.message)
_invalid(message) = throw(TranslationFailure(MOI.INVALID_MODEL, message))

# A CachingOptimizer maps source indices before it forwards a result query,
# even when ResultCount is zero. Always finish this inexpensive index-only pass.
function _index_map(src, variables=MOI.get(src, MOI.ListOfVariableIndices()), bound_owners=Dict{DataType,BitSet}())
    indices = MOI.Utilities.IndexMap()
    for (i, vi) in enumerate(variables)
        indices[vi] = MOI.VariableIndex(i)
    end
    for (F, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent())
        F === MOI.VariableIndex && S <: BoundSet || throw(MOI.UnsupportedConstraint{F,S}())
        owners = get!(BitSet, bound_owners, S)
        for ci in MOI.get(src, MOI.ListOfConstraintIndices{F,S}())
            owner = MOI.VariableIndex(ci.value)
            id = indices[owner].value
            indices[ci] = MOI.ConstraintIndex{F,S}(id)
            push!(owners, id)
        end
    end
    return indices
end

# Check on entry to each loop and between small batches of terms or coordinates.
@inline function _translation_checkpoint(checkpoint, ordinal::Int)
    if (ordinal - 1) % 256 == 0
        checkpoint()
    end
    return
end

function _check_source_attributes(src)
    for attr in MOI.get(src, MOI.ListOfModelAttributesSet())
        attr isa Union{MOI.Name,MOI.ObjectiveSense,MOI.ObjectiveFunction} ||
            throw(MOI.UnsupportedAttribute(attr))
    end
    variable_attributes = MOI.get(src, MOI.ListOfVariableAttributesSet())
    for attr in variable_attributes
        attr isa MOI.VariableName || throw(MOI.UnsupportedAttribute(attr))
    end
    for (F, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent())
        for attr in MOI.get(src, MOI.ListOfConstraintAttributesSet{F,S}())
            attr isa MOI.ConstraintName || throw(MOI.UnsupportedAttribute(attr))
        end
    end
    return MOI.VariableName() in variable_attributes
end

_endpoints(s::MOI.LessThan) = (-Inf, s.upper)
_endpoints(s::MOI.GreaterThan) = (s.lower, Inf)
_endpoints(s::MOI.EqualTo) = (s.value, s.value)
_endpoints(s::MOI.Interval) = (s.lower, s.upper)

function _variable_label(src, vi, has_names)
    name = has_names ? MOI.get(src, MOI.VariableName(), vi) : ""
    return isempty(name) ? string(vi) : "$vi ($name)"
end

function _box(src, indices, variables, has_names, checkpoint)
    lower, upper = fill(-Inf, length(variables)), fill(Inf, length(variables))
    for (F, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent())
        for (k, ci) in enumerate(MOI.get(src, MOI.ListOfConstraintIndices{F,S}()))
            _translation_checkpoint(checkpoint, k)
            lo, hi = _endpoints(MOI.get(src, MOI.ConstraintSet(), ci))
            i = indices[ci].value
            if !(lo < Inf && hi > -Inf)
                label = _variable_label(src, MOI.VariableIndex(ci.value), has_names)
                _invalid("invalid endpoint in bound $ci on $label")
            end
            lower[i] = max(lower[i], lo)
            upper[i] = min(upper[i], hi)
        end
    end
    incomplete = nothing
    for (k, vi) in enumerate(variables)
        _translation_checkpoint(checkpoint, k)
        i = indices[vi].value
        if lower[i] > upper[i]
            label = _variable_label(src, vi, has_names)
            throw(TranslationFailure(MOI.INFEASIBLE, "empty bound intersection on $label"))
        elseif isnothing(incomplete) && (!isfinite(lower[i]) || !isfinite(upper[i]))
            incomplete = vi
        end
    end
    if !isnothing(incomplete)
        label = _variable_label(src, incomplete, has_names)
        _invalid("finite lower and upper bounds required for $label")
    end
    return lower, upper
end

function _checked_sum(total)
    value = Float64(total)
    isfinite(value) && QHDOPT._exact(value) == total ||
        _invalid("aggregated objective coefficient is not exactly representable in Float64")
    return value
end

function _term_coefficient(value)
    isfinite(value) || _invalid("nonfinite objective coefficient $value")
    return QHDOPT._exact(value)
end

function _objective_coordinate(indices, vi)
    haskey(indices, vi) || _invalid("objective uses unknown variable $vi")
    return indices[vi].value
end

function _affine_sums(terms, indices, checkpoint)
    sums = Dict{Int,Rational{BigInt}}()
    for (k, term) in enumerate(terms)
        _translation_checkpoint(checkpoint, k)
        i = _objective_coordinate(indices, term.variable)
        sums[i] = get(sums, i, 0) + _term_coefficient(term.coefficient)
    end
    return sums
end

function _quadratic_sums(terms, indices, checkpoint)
    entries = Dict{Tuple{Int,Int},Rational{BigInt}}()
    for (k, term) in enumerate(terms)
        _translation_checkpoint(checkpoint, k)
        key = minmax(_objective_coordinate(indices, term.variable_1), _objective_coordinate(indices, term.variable_2))
        entries[key] = get(entries, key, 0) + _term_coefficient(term.coefficient)
    end
    return entries
end

# A raw diagonal term contributes t*x_i^2/2, so H_ii receives t unchanged.
# Group both orientations before mirroring so sparse assembly never sums in Float64.
function _quadratic_matrix(terms, indices, n, checkpoint)
    entries = _quadratic_sums(terms, indices, checkpoint)
    rows, cols, values = Int[], Int[], Float64[]
    for (k, ((i, j), total)) in enumerate(entries)
        _translation_checkpoint(checkpoint, k)
        iszero(total) && continue
        value = _checked_sum(total)
        push!(rows, i)
        push!(cols, j)
        push!(values, value)
        if i != j
            push!(rows, j)
            push!(cols, i)
            push!(values, value)
        end
    end
    checkpoint()
    return sparse(rows, cols, values, n, n)
end

function _objective_data(src, indices, sense, n, checkpoint)
    b = zeros(n)
    sense == MOI.FEASIBILITY_SENSE && return spzeros(n, n), b, 0.0
    F = MOI.get(src, MOI.ObjectiveFunctionType())
    attr = MOI.ObjectiveFunction{F}()
    F <: PolynomialObjective || throw(MOI.UnsupportedAttribute(attr))
    f = MOI.get(src, attr)
    if f isa MOI.VariableIndex
        b[_objective_coordinate(indices, f)] = 1.0
        return spzeros(n, n), b, 0.0
    end
    isfinite(f.constant) || _invalid("nonfinite objective constant $(f.constant)")
    if f isa MOI.ScalarQuadraticFunction
        H = _quadratic_matrix(f.quadratic_terms, indices, n, checkpoint)
        affine = f.affine_terms
    else
        H = spzeros(n, n)
        affine = f.terms
    end
    sums = _affine_sums(affine, indices, checkpoint)
    for (k, (i, total)) in enumerate(sums)
        _translation_checkpoint(checkpoint, k)
        b[i] = _checked_sum(total)
    end
    return H, b, f.constant
end

include("jump_nonlinear.jl")

function _translate(src, indices, variables=MOI.get(src, MOI.ListOfVariableIndices()), checkpoint=()->nothing)
    checkpoint()
    has_names = _check_source_attributes(src)
    isempty(variables) && _invalid("the model must contain at least one variable")
    sense = MOI.get(src, MOI.ObjectiveSense())
    checkpoint()
    if sense != MOI.FEASIBILITY_SENSE && MOI.get(src, MOI.ObjectiveFunctionType()) == MOI.ScalarNonlinearFunction
        lower, upper = _box(src, indices, variables, has_names, checkpoint)
        f = MOI.get(src, MOI.ObjectiveFunction{MOI.ScalarNonlinearFunction}())
        return _structured_model(f, indices, sense, lower, upper, checkpoint), sense
    end
    H, b, constant = _objective_data(src, indices, sense, length(variables), checkpoint)
    checkpoint()
    lower, upper = _box(src, indices, variables, has_names, checkpoint)
    if sense == MOI.MAX_SENSE
        H, b, constant = -H, -b, -constant
    end
    checkpoint()
    return QHDOPT.quadratic_problem(H, b; constant, lower, upper), sense
end

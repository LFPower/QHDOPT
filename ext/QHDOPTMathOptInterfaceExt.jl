module QHDOPTMathOptInterfaceExt

import MathOptInterface as MOI
import QHDOPT
using SparseArrays: sparse, spzeros

include("jump_translation.jl")

Base.@kwdef mutable struct Options
    encoding::Union{Nothing,QHDOPT.GridEncoding} = nothing
    validity::Union{Nothing,QHDOPT.BlockValidity} = nothing
    backend::Union{Nothing,QHDOPT.PythonQIHDBackend} = nothing
    sampler::QHDOPT.QIHDConfig = QHDOPT.QIHDConfig()
    storage::Symbol = :sparse
    refiner::QHDOPT.AbstractRefiner = QHDOPT.NoRefiner()
    execution::QHDOPT.MultistartExecution = QHDOPT.SerialMultistart()
end

struct Outcome
    status::MOI.TerminationStatusCode
    raw_status::String
    native::Union{Nothing,QHDOPT.QHDResult}
    point::Union{Nothing,Vector{Float64}}
    objective::Union{Nothing,Float64}
end
Outcome(status, message) = Outcome(status, message, nothing, nothing, nothing)

struct Snapshot
    outcome::Outcome
    variable_count::Int
    bound_owners::Dict{DataType,BitSet}
    seconds::Float64
end

mutable struct Optimizer <: MOI.AbstractOptimizer
    options::Options
    time_limit::Union{Nothing,Float64}
    set_attributes::Set{MOI.AbstractOptimizerAttribute}
    snapshot::Union{Nothing,Snapshot}
end

function QHDOPT.Optimizer(; kwargs...)
    opt = Optimizer(Options(), nothing, Set{MOI.AbstractOptimizerAttribute}(), nothing)
    for (name, value) in pairs(kwargs)
        MOI.set(opt, MOI.RawOptimizerAttribute(String(name)), value)
    end
    return opt
end

MOI.get(::Optimizer, ::MOI.SolverName) = "QHDOPT"
MOI.get(::Optimizer, ::MOI.SolverVersion) = string(pkgversion(QHDOPT))
MOI.supports_incremental_interface(::Optimizer) = false
MOI.is_empty(opt::Optimizer) = isnothing(opt.snapshot)
function MOI.empty!(opt::Optimizer)
    opt.snapshot = nothing
    return
end

function MOI.copy_to(opt::Optimizer, source::MOI.ModelLike)
    # CachingOptimizer attaches via copy_to instead of our one-shot solve when a callback is set.
    for attr in MOI.get(source, MOI.ListOfModelAttributesSet())
        if attr isa MOI.AbstractCallback
            MOI.empty!(opt)
            throw(MOI.UnsupportedAttribute(attr))
        end
    end
    return invoke(MOI.copy_to, Tuple{MOI.ModelLike,MOI.ModelLike}, opt, source)
end

MOI.supports_constraint(::Optimizer, ::Type{MOI.VariableIndex}, ::Type{<:BoundSet}) = true
MOI.supports(::Optimizer, ::MOI.ObjectiveSense) = true
MOI.supports(::Optimizer, ::MOI.ObjectiveFunction{F}) where {F<:ObjectiveFunction} = true
MOI.get(::Optimizer, ::MOI.ListOfSupportedNonlinearOperators) = collect(StructuredOperators)

MOI.get(opt::Optimizer, ::MOI.ListOfOptimizerAttributesSet) = collect(opt.set_attributes)
function _record_option!(opt, attr, value)
    if isnothing(value)
        delete!(opt.set_attributes, attr)
    else
        push!(opt.set_attributes, attr)
    end
    return
end

MOI.supports(::Optimizer, ::MOI.TimeLimitSec) = true
MOI.get(opt::Optimizer, ::MOI.TimeLimitSec) = opt.time_limit
function MOI.set(opt::Optimizer, attr::MOI.TimeLimitSec, limit::Union{Nothing,Real})
    if !isnothing(limit) && (limit isa Bool || !isfinite(Float64(limit)) || limit < 0)
        throw(MOI.SetAttributeNotAllowed(attr, "time limit must be finite and nonnegative"))
    end
    opt.time_limit = limit
    _record_option!(opt, attr, limit)
    return
end

function MOI.supports(::Optimizer, attr::MOI.RawOptimizerAttribute)
    return hasfield(Options, Symbol(attr.name))
end

function MOI.get(opt::Optimizer, attr::MOI.RawOptimizerAttribute)
    MOI.supports(opt, attr) || throw(MOI.UnsupportedAttribute(attr))
    return deepcopy(getfield(opt.options, Symbol(attr.name)))
end

function MOI.set(opt::Optimizer, attr::MOI.RawOptimizerAttribute, value)
    MOI.supports(opt, attr) || throw(MOI.UnsupportedAttribute(attr))
    name = Symbol(attr.name)
    T = fieldtype(Options, name)
    value isa T || throw(MOI.SetAttributeNotAllowed(attr, "expected $T, got $(typeof(value))"))
    setfield!(opt.options, name, deepcopy(value))
    _record_option!(opt, attr, value)
    return
end

function _configuration_error(options::Options)
    if !(options.encoding isa Union{QHDOPT.Unary,QHDOPT.OneHot,QHDOPT.Hamming})
        return "choose an explicit Unary, OneHot or Hamming encoding"
    elseif options.encoding isa QHDOPT.Hamming && !isnothing(options.validity)
        return "Hamming encoding requires validity=nothing"
    elseif options.encoding isa QHDOPT.TableEncoding && isnothing(options.validity)
        return "Unary and OneHot encodings require an explicit validity policy"
    elseif !(options.storage in (:sparse, :dense))
        return "storage must be :sparse or :dense"
    end
    return nothing
end

function _native_outcome(model, sense, native::QHDOPT.QHDResult)
    if native.status == :deterministic && native.provenance.shortcut in (:all_fixed, :constant)
        status = MOI.OPTIMAL
    elseif native.status == :deadline
        status = MOI.TIME_LIMIT
    elseif native.status in (:coarse, :refined, :all_invalid, :empty)
        status = MOI.OTHER_LIMIT
    else
        return Outcome(MOI.OTHER_ERROR, "unexpected sampled solve status: $(native.status)",
                       native, nothing, nothing)
    end
    raw = string(native.status)
    selected = QHDOPT.solution(native)
    if isnothing(selected) && native.status in (:all_invalid, :empty, :deadline)
        return Outcome(status, raw, native, nothing, nothing)
    end
    # Validate and reevaluate in original coordinates using the same checks as native refinement.
    point, score, validation, reason = QHDOPT._validate_return(model, selected, 0.0)
    if isnothing(point)
        return Outcome(MOI.NUMERICAL_ERROR, "validation: $validation $reason",
                       native, nothing, nothing)
    end
    objective = sense == MOI.MAX_SENSE ? -score : score
    return Outcome(status, raw, native, point, objective)
end

function _solve_snapshot(opt, source, index_map, variables, clock)
    checkpoint = () -> QHDOPT._check_time(clock, :translation)
    model, sense = _translate(source, index_map, variables, checkpoint)
    remaining = QHDOPT._check_time(clock, :translation)
    options = opt.options
    issue = _configuration_error(options)
    isnothing(issue) || return Outcome(MOI.INVALID_OPTION, "configuration: $issue")
    native = try
        QHDOPT._solve_classified(model; options.encoding, options.validity, options.storage,
            options.backend, options.sampler, options.refiner, options.execution,
            time_limit_seconds=isnothing(clock.limit) ? nothing : remaining)
    catch err
        if err isa QHDOPT._SolveFailure
            status = if err.kind == :invalid_option
                MOI.INVALID_OPTION
            elseif err.kind == :invalid_model
                MOI.INVALID_MODEL
            elseif err.kind == :numerical_admission
                MOI.NUMERICAL_ERROR
            else
                MOI.OTHER_ERROR
            end
            return Outcome(status, sprint(showerror, err))
        elseif err isa Union{QHDOPT.SolveDeadlineError,InterruptException,OutOfMemoryError}
            rethrow()
        end
        return Outcome(MOI.OTHER_ERROR, "execution: $(sprint(showerror, err))")
    end
    return _native_outcome(model, sense, native)
end

function MOI.optimize!(opt::Optimizer, source::MOI.ModelLike)
    MOI.empty!(opt)
    clock = QHDOPT.SolveClock(opt.time_limit)
    variables = MOI.get(source, MOI.ListOfVariableIndices())
    bound_owners = Dict{DataType,BitSet}()
    index_map = _index_map(source, variables, bound_owners)
    outcome = try
        _solve_snapshot(opt, source, index_map, variables, clock)
    catch err
        if err isa TranslationFailure
            Outcome(err.status, "translation: $(err.message)")
        elseif err isa QHDOPT.SolveDeadlineError
            Outcome(MOI.TIME_LIMIT, sprint(showerror, err))
        else
            rethrow()
        end
    end
    opt.snapshot = Snapshot(outcome, length(variables), bound_owners, QHDOPT._elapsed(clock.started))
    return index_map, false
end

function MOI.get(opt::Optimizer, ::MOI.TerminationStatus)
    return isnothing(opt.snapshot) ? MOI.OPTIMIZE_NOT_CALLED : opt.snapshot.outcome.status
end
function MOI.get(opt::Optimizer, ::MOI.RawStatusString)
    return isnothing(opt.snapshot) ? "optimize! has not been called" : opt.snapshot.outcome.raw_status
end
function MOI.get(opt::Optimizer, ::MOI.ResultCount)
    return Int(!isnothing(opt.snapshot) && !isnothing(opt.snapshot.outcome.point))
end
function MOI.get(opt::Optimizer, ::MOI.SolveTimeSec)
    return isnothing(opt.snapshot) ? 0.0 : opt.snapshot.seconds
end
function MOI.get(opt::Optimizer, attr::MOI.PrimalStatus)
    return 1 <= attr.result_index <= MOI.get(opt, MOI.ResultCount()) ? MOI.FEASIBLE_POINT : MOI.NO_SOLUTION
end
MOI.get(::Optimizer, ::MOI.DualStatus) = MOI.NO_SOLUTION
function MOI.get(opt::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(opt, attr)
    return opt.snapshot.outcome.objective::Float64
end
function MOI.is_valid(opt::Optimizer, index::MOI.VariableIndex)
    return !isnothing(opt.snapshot) && 1 <= index.value <= opt.snapshot.variable_count
end
function MOI.is_valid(opt::Optimizer, index::MOI.ConstraintIndex{F,S}) where {F,S}
    isnothing(opt.snapshot) && return false
    owners = get(opt.snapshot.bound_owners, S, nothing)
    return F == MOI.VariableIndex && !isnothing(owners) && index.value in owners
end
function MOI.get(opt::Optimizer, attr::MOI.VariablePrimal, index::MOI.VariableIndex)
    MOI.check_result_index_bounds(opt, attr)
    MOI.is_valid(opt, index) || throw(MOI.InvalidIndex(index))
    return opt.snapshot.outcome.point[index.value]
end
function MOI.get(opt::Optimizer, attr::MOI.ConstraintPrimal, index::MOI.ConstraintIndex{MOI.VariableIndex,<:BoundSet})
    MOI.check_result_index_bounds(opt, attr)
    MOI.is_valid(opt, index) || throw(MOI.InvalidIndex(index))
    return opt.snapshot.outcome.point[index.value]
end

struct NativeResultAttribute <: MOI.AbstractModelAttribute end
QHDOPT.NativeResult() = NativeResultAttribute()
MOI.is_set_by_optimize(::NativeResultAttribute) = true
MOI.attribute_value_type(::NativeResultAttribute) = Union{Nothing,QHDOPT.QHDResult}
function MOI.get(opt::Optimizer, ::NativeResultAttribute)
    return isnothing(opt.snapshot) ? nothing : deepcopy(opt.snapshot.outcome.native)
end
# Native solver settings and results use QHDOPT coordinates, independent of MOI variable bridges.
const NativeOption = Union{QHDOPT.GridEncoding,QHDOPT.BlockValidity,QHDOPT.PythonQIHDBackend,
                           QHDOPT.QIHDConfig,QHDOPT.AbstractRefiner,QHDOPT.MultistartExecution}
MOI.Bridges.bridged_function(::MOI.Bridges.AbstractBridgeOptimizer, option::NativeOption) = option
MOI.Bridges.unbridged_function(::MOI.Bridges.AbstractBridgeOptimizer, value::Union{NativeOption,QHDOPT.QHDResult}) = value
MOI.Utilities.map_indices(_, ::NativeResultAttribute, result::Union{Nothing,QHDOPT.QHDResult}) = result

end # module

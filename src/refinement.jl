abstract type AbstractRefiner end
struct NoRefiner <: AbstractRefiner end

struct StartID
    origin::Symbol
    ordinal::Int
end

abstract type AbstractStart end

mutable struct DecodedStart <: AbstractStart
    const start_id::StartID
    const key::Tuple{Vararg{Int}}
    const point::Vector{Float64}
    const value::Float64
    const first_trial::Int
    multiplicity::Int
end

struct SuppliedStart <: AbstractStart
    start_id::StartID
    point::Vector{Float64}
    value::Float64
    input_ordinal::Int
end

struct CandidateSelection
    origin::Symbol
    start_id::Union{Nothing,StartID}
    point::Vector{Float64}
    value::Float64
end

# The adapter reports what the solver actually did. Feasibility, objective
# reevaluation and candidate selection are shared by every start and adapter.
Base.@kwdef struct RefinerOutcome
    status::Symbol
    solver_called::Bool
    raw_status::Union{Nothing,Int,String,Symbol} = nothing
    point::Union{Nothing,Vector{Float64}} = nothing
    message::String = ""
    diagnostics::NamedTuple = (;)
end

struct RefinementRecord
    start_id::StartID
    solver::Symbol
    options::NamedTuple
    status::Symbol
    solver_called::Bool
    raw_status::Union{Nothing,Int,String,Symbol}
    returned_point::Union{Nothing,Vector{Float64}}
    accepted_point::Union{Nothing,Vector{Float64}}
    returned_value::Union{Nothing,Float64}
    validation::Symbol
    usable::Bool
    message::String
    elapsed_seconds::Float64
    diagnostics::NamedTuple
end

_refiner_info(r::AbstractRefiner) = (; solver=nameof(typeof(r)), options=(;), feasibility_tolerance=0.0)
_refiner_info(::NoRefiner) = (; solver=:none, options=(;), feasibility_tolerance=0.0)
_refiner_info(r::AbstractRefiner, ::QHDProblem) = _refiner_info(r)
_elapsed(start_ns) = (time_ns() - start_ns) / 1e9

"""A cooperative solve deadline expired before the named stage could finish."""
struct SolveDeadlineError <: Exception
    stage::Symbol
end
Base.showerror(io::IO, e::SolveDeadlineError) = print(io, "solve deadline reached during ", e.stage)

struct SolveClock
    started::UInt64
    limit::Union{Float64,Nothing}
    function SolveClock(limit)
        if limit !== nothing
            limit isa Real && !(limit isa Bool) && isfinite(Float64(limit)) && limit >= 0 ||
                throw(ArgumentError("time_limit_seconds must be finite and nonnegative"))
        end
        new(time_ns(), limit)
    end
end
_remaining(clock::SolveClock) = isnothing(clock.limit) ? Inf : clock.limit - _elapsed(clock.started)
function _check_time(clock, stage)
    remaining = _remaining(clock)
    remaining > 0 || throw(SolveDeadlineError(stage))
    return remaining
end

function _validate_return(model, x, tol)
    isnothing(x) && return (nothing, nothing, :no_return, "")
    length(x) == length(model.names) || return (nothing, nothing, :wrong_dimension, "wrong return dimension")
    all(isfinite, x) || return (nothing, nothing, :nonfinite, "nonfinite returned coordinate")
    for i in eachindex(x)
        if model.lower[i] - x[i] > tol || x[i] - model.upper[i] > tol
            return (nothing, nothing, :infeasible, "coordinate $i is outside the original box")
        end
    end
    accepted = clamp.(x, model.lower, model.upper)
    value = try
        objective_value(model, accepted)
    catch err
        err isa Union{ArgumentError,DomainError,OverflowError,InexactError} || rethrow()
        return (nothing, nothing, :invalid_objective, sprint(showerror, err))
    end
    isfinite(value) || return (nothing, nothing, :invalid_objective, "nonfinite original objective")
    return accepted, value, accepted == x ? :valid : :projected, ""
end

# Validate in the parent using the same owned model regardless of where the
# native call ran. A worker's return status and its point's usability differ.
function _refinement_record(model, start, info, outcome, seconds; worker=nothing)
    validating = time_ns()
    raw = isnothing(outcome.point) ? nothing : copy(outcome.point)
    !outcome.solver_called && !isnothing(raw) && error("unattempted refinement reported a returned point")
    accepted, value, validation, reason = _validate_return(model, raw, info.feasibility_tolerance)
    message = join(filter(!isempty, (outcome.message, reason)), ". ")
    diagnostics = deepcopy(outcome.diagnostics)
    isnothing(worker) || (diagnostics = merge(diagnostics, (; worker)))
    return RefinementRecord(start.start_id, info.solver, info.options, outcome.status,
                            outcome.solver_called, outcome.raw_status, raw, accepted, value, validation,
                            !isnothing(accepted), message, seconds + _elapsed(validating), diagnostics)
end

function _refine_population(refiner, model, starts, info, clock; structural=false)
    records = RefinementRecord[]
    for start in starts
        started = time_ns()
        outcome = if refiner isa NoRefiner
            RefinerOutcome(status=:not_requested, solver_called=false)
        elseif _remaining(clock) <= 0
            RefinerOutcome(status=:not_attempted_deadline, solver_called=false)
        elseif structural
            RefinerOutcome(status=:not_attempted_structural, solver_called=false)
        else
            _run_refiner(refiner, model, start)
        end
        push!(records, _refinement_record(model, start, info, outcome, _elapsed(started)))
    end
    return records
end

function _stage_candidate(start, record, stage)
    origin = start.start_id.origin
    if stage == :refined || (stage == :selected && record.usable && record.returned_value < start.value)
        record.usable || return nothing
        return CandidateSelection(:refined, start.start_id, record.accepted_point, record.returned_value)
    elseif stage == :selected || (stage == :coarse && origin == :sampled) || (stage == :initial && origin == :classical)
        return CandidateSelection(origin, start.start_id, start.point, start.value)
    end
    return nothing
end

function _best_stage(starts, records, stage)
    best = nothing
    for (start, record) in zip(starts, records)
        candidate = _stage_candidate(start, record, stage)
        if !isnothing(candidate) && (isnothing(best) || candidate.value < best.value)
            best = candidate
        end
    end
    return best
end

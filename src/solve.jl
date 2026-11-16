"""The isolated OpenPhiSolve CPU worker and its process timeout, in seconds."""
struct PythonQIHDBackend
    python::String
    timeout_seconds::Float64

    function PythonQIHDBackend(; python::AbstractString, timeout_seconds::Real=120)
        isempty(strip(python)) && throw(ArgumentError("python must name an interpreter"))
        t = Float64(timeout_seconds)
        !(timeout_seconds isa Bool) && isfinite(t) && t > 0 ||
            throw(ArgumentError("timeout_seconds must be finite and positive"))
        new(python, t)
    end
end

struct TrialRecord
    status::Symbol
    start_id::Union{Nothing,StartID}
    # Spin coordinate for an invalid spin, free-variable block for an invalid codeword.
    offending_index::Union{Nothing,Int}
    encoded_energy::Union{Nothing,Float64}
    penalty_energy::Union{Nothing,Float64}
end

"""An owned solve snapshot. Sampled trials and classical input ordinals are separate populations."""
struct QHDResult
    status::Symbol
    names::Vector{Symbol}
    raw::Union{Nothing,SampleResult}
    trial_map::Vector{TrialRecord}
    sampled_starts::Vector{DecodedStart}
    supplied_starts::Vector{SuppliedStart}
    refinement_records::Vector{RefinementRecord}
    attempt_map::Dict{StartID,Int}
    selected::Union{Nothing,CandidateSelection}
    counts::NamedTuple
    provenance::NamedTuple
    timings::NamedTuple
end

function _provenance(p, backend, sampler, info, clock)
    return (; model_fingerprint=p.model_fingerprint, compiled_fingerprint=p.fingerprint,
            encoding=_encoding_id(p.encoding.spec), grid=_grid_convention(p.encoding.spec),
            resolutions=copy(p.encoding.resolutions), validity=p.encoding.validity,
            storage=p.storage, spin_offset=p.spin_offset, shortcut=p.shortcut,
            lower=copy(p.model.lower), upper=copy(p.model.upper),
            backend, requested_sampler=sampler, refiner=info.solver, refiner_details=info,
            time_limit_seconds=clock.limit)
end

function _counts(requested, trials, starts, supplied, records)
    invalid_spin = count(t -> t.status == :invalid_spin, trials)
    invalid_codeword = count(t -> t.status == :invalid_codeword, trials)
    valid = sum(st -> st.multiplicity, starts; init=0)
    actual = length(trials)
    actual == invalid_spin + invalid_codeword + valid || error("incomplete trial accounting")
    calls = count(r -> r.solver_called, records)
    attempted = count(r -> r.start_id.origin == :classical && r.solver_called, records)
    usable = count(r -> r.start_id.origin == :classical && r.usable, records)
    return (; requested_trials=requested, actual_trials=actual, invalid_spin, invalid_codeword,
            valid, unique_starts=length(starts), refinement_calls=calls,
            classical_requested=length(supplied), classical_attempted=attempted, classical_completed=attempted,
            classical_usable_returned=usable, classical_not_attempted=length(supplied)-attempted)
end

_decoding_error(err) = throw(_SolveFailure(:numerical_admission, :decoding, err))

function _decoded_objective_value(objective, point)
    try
        return objective_value(objective, point)
    catch err
        err isa Union{ArgumentError,DomainError,OverflowError,InexactError} || rethrow()
        _decoding_error(err)
    end
end

function _spin_energy(p, spins)
    prob = p.sampling_problem
    # Sum the stored coefficients before rounding. Direct Float64 dot products
    # can discard a small objective between the fields and the external offset.
    objective = QuadraticObjective(prob.Q, prob.w, p.spin_offset)
    return _decoded_objective_value(objective, spins)
end

function _decode_trials(p, samples)
    resolutions = p.encoding.resolutions
    m = sum(resolutions)
    size(samples, 2) == m || throw(DimensionMismatch("sample width must equal the $m encoded slots"))
    starts = DecodedStart[]
    trials = TrialRecord[]
    ordinals = Dict{Tuple{Vararg{Int}},Int}()
    levels = zeros(Int, length(resolutions))
    for (trial, spins) in enumerate(eachrow(samples))
        bad_spin = findfirst(s -> s != -1 && s != 1, spins)
        if bad_spin !== nothing
            push!(trials, TrialRecord(:invalid_spin, nothing, bad_spin, nothing, nothing))
            continue
        end
        bits = spins .== -1  # q = (1-s)/2
        first_bad_block = nothing
        violations = 0
        for (i, r) in enumerate(resolutions)
            offset = p.encoding.offsets[i]
            units = _violation(p.encoding.spec, @view bits[offset+1:offset+r])
            if units > 0
                violations += units
                isnothing(first_bad_block) && (first_bad_block = i)
            end
        end
        if first_bad_block !== nothing
            policy = p.encoding.validity
            penalty = policy isa FoldedPenalty ?
                _coefficient(_exact(policy.weight) * violations, "encoding penalty at trial $trial", _decoding_error) : 0.0
            push!(trials, TrialRecord(:invalid_codeword, nothing, first_bad_block,
                                      _spin_energy(p, spins), penalty))
            continue
        end
        for (i, r) in enumerate(resolutions)
            offset = p.encoding.offsets[i]
            levels[i] = _decode_level(p.encoding.spec, @view bits[offset+1:offset+r])
        end
        key = Tuple(levels)
        ordinal = get(ordinals, key, 0)
        if ordinal == 0
            point = reconstruct(p.transform, levels .// resolutions)
            value = _decoded_objective_value(p.model, point)
            ordinal = length(starts) + 1
            ordinals[key] = ordinal
            push!(starts, DecodedStart(StartID(:sampled, ordinal), key, point, value, trial, 1))
            energy = _spin_energy(p, spins)
        else
            starts[ordinal].multiplicity += 1
            energy = trials[starts[ordinal].first_trial].encoded_energy
        end
        push!(trials, TrialRecord(:valid, starts[ordinal].start_id, nothing, energy, 0.0))
    end
    return trials, starts
end

function _finish_result(model, raw, trials, sampled, supplied, requested, provenance, refiner, execution,
                        clock, phases; deterministic=nothing, structural=false)
    starts = isempty(supplied) ? sampled : supplied
    refine_started = time_ns()
    records, execution_info = _execute_population(execution, refiner, model, starts,
                                                  provenance.refiner_details, clock; structural)
    provenance = merge(provenance, (; execution=execution_info))
    refinement_seconds = refiner isa NoRefiner || isempty(starts) ? 0.0 : _elapsed(refine_started)
    attempts = Dict(record.start_id => i for (i, record) in enumerate(records))
    counts = _counts(requested, trials, sampled, supplied, records)
    selected = isnothing(deterministic) ? _best_stage(starts, records, :selected) : deterministic
    status = if !isnothing(deterministic)
        :deterministic
    elseif _remaining(clock) <= 0
        :deadline
    elseif isempty(starts)
        isempty(trials) ? :empty : :all_invalid
    elseif any(r -> r.usable, records)
        :refined
    else
        isempty(supplied) ? :coarse : :initial
    end
    timings = (; phases..., refinement_seconds, total_seconds=_elapsed(clock.started))
    return QHDResult(status, copy(model.names), raw, trials, sampled, supplied, records, attempts,
                     deepcopy(selected), counts, provenance, timings)
end

function _finish_sampled(p, raw, requested, provenance, refiner, execution, clock, compilation, preparation)
    decoding_started = time_ns()
    # Complete the trial map for every committed row, including after a deadline.
    trials, starts = _decode_trials(p, raw.samples)
    phases = (; compilation_seconds=compilation, preparation_seconds=preparation,
              decoding_seconds=_elapsed(decoding_started))
    return _finish_result(p.model, raw, trials, starts, SuppliedStart[], requested,
                          provenance, refiner, execution, clock, phases)
end

# Also used to verify the decoder against committed row fixtures independently of stochastic sampling.
function coarse_result(p::CompiledQHD, raw::SampleResult; requested_trials::Int=size(raw.samples, 1),
                       refiner::AbstractRefiner=NoRefiner(), execution::MultistartExecution=SerialMultistart(),
                       time_limit_seconds=nothing)
    clock = SolveClock(time_limit_seconds)
    p = deepcopy(p)
    _validate_artifact(p)
    p.shortcut == :none || throw(ArgumentError("a deterministic artifact has no sampled rows"))
    requested_trials >= 0 || throw(ArgumentError("requested_trials must be nonnegative"))
    owned = deepcopy(raw)
    return _native_errors() do
        prov = _provenance(p, nothing, nothing, _refiner_info(refiner, p.model), clock)
        _finish_sampled(p, owned, requested_trials, prov, refiner, execution, clock, 0.0, _elapsed(clock.started))
    end
end

function _solve(p, backend, sampler, refiner, execution, clock, compilation)
    _check_time(clock, :preparation)
    _validate_artifact(p)
    provenance = _provenance(p, backend, sampler, _refiner_info(refiner, p.model), clock)
    if p.shortcut != :none
        point = _shortcut_point(p.transform, p.encoding)
        candidate = CandidateSelection(p.shortcut, nothing, point, objective_value(p.model, point))
        phases = (; compilation_seconds=compilation, preparation_seconds=_elapsed(clock.started)-compilation,
                  decoding_seconds=0.0)
        return _finish_result(p.model, nothing, TrialRecord[], DecodedStart[], SuppliedStart[],
                              sampler.n_shots, provenance, refiner, execution, clock, phases; deterministic=candidate)
    end
    isnothing(backend) && _invalid_option(:sampling, ArgumentError("a sampled solve requires PythonQIHDBackend(python=...)"))
    job = SamplingJob(p.sampling_problem, sampler)
    preparation = _elapsed(clock.started) - compilation
    _check_time(clock, :sampling)
    raw = only(_sample_jobs([job]; backend.python, backend.timeout_seconds, preflight=true,
                            remaining_time=() -> _check_time(clock, :sampling)))
    return _finish_sampled(p, raw, sampler.n_shots, provenance, refiner, execution, clock, compilation, preparation)
end

"""
    solve(prepared; backend, sampler=QIHDConfig(), refiner=NoRefiner(), execution=SerialMultistart(), time_limit_seconds=nothing)
    solve(model; encoding, validity, storage=:sparse, backend, sampler, refiner)

Sample a prepared bounded objective and optionally refine each unique decoded start in original coordinates.
All-fixed and algebraically constant models return a deterministic candidate with zero
trials and need no backend. Worker failures throw `WorkerError`.
"""
function solve(p::CompiledQHD; backend::Union{Nothing,PythonQIHDBackend}=nothing,
               sampler::QIHDConfig=QIHDConfig(), refiner::AbstractRefiner=NoRefiner(),
               execution::MultistartExecution=SerialMultistart(), time_limit_seconds=nothing)
    clock = SolveClock(time_limit_seconds)
    _check_time(clock, :preparation)
    # Keep the coefficients sent to Python and the model used after its return
    # in the same run-owned snapshot, including during the process wait.
    return _native_errors(() -> _solve(deepcopy(p), backend, sampler, refiner, execution, clock, 0.0))
end

function solve(model::QHDProblem; kwargs...)
    return _native_errors(() -> _solve_classified(model; kwargs...))
end

function _solve_classified(model::QHDProblem; encoding::GridEncoding, validity=_default_validity(encoding),
                          storage::Symbol=:sparse, backend::Union{Nothing,PythonQIHDBackend}=nothing,
                          sampler::QIHDConfig=QIHDConfig(), refiner::AbstractRefiner=NoRefiner(),
                          execution::MultistartExecution=SerialMultistart(), time_limit_seconds=nothing)
    clock = SolveClock(time_limit_seconds)
    _check_time(clock, :compilation)
    p = _compile(model, encoding, validity, storage, () -> _check_time(clock, :compilation))
    return _solve(p, backend, sampler, refiner, execution, clock, _elapsed(clock.started))
end

function _supplied_start(model, input, ordinal)
    label = "supplied start $ordinal"
    input isa AbstractVector{<:Real} && length(input) == length(model.names) ||
        throw(ArgumentError("$label must be a real vector with $(length(model.names)) coordinates"))
    point = Vector{Float64}(undef, length(input))
    for (i, x) in enumerate(input)
        # Check membership before conversion, including exact fixed coordinates.
        isfinite(x) && model.lower[i] <= x <= model.upper[i] ||
            throw(ArgumentError("$label coordinate $i is nonfinite or outside its original bounds"))
        point[i] = _coefficient(x, "$label coordinate $i")
    end
    value = try
        objective_value(model, point)
    catch err
        err isa Union{ArgumentError,DomainError,OverflowError,InexactError} || rethrow()
        throw(ArgumentError("original objective at $label: $(sprint(showerror, err))"))
    end
    isfinite(value) || throw(ArgumentError("nonfinite original objective at $label"))
    return SuppliedStart(StartID(:classical, ordinal), point, value, ordinal)
end

function _structural_classical(model::QHDProblem{QuadraticObjective})
    free = findall(model.lower .< model.upper)
    return isempty(free) || (!_has_nonzero(model.objective.H[free, free]) &&
           all(iszero, _reduced_gradient(model, _exact.(model.lower))[free]))
end
function _structural_classical(model::QHDProblem{StructuredObjective})
    return model.lower == model.upper || isempty(_coordinates(_substitute(model.objective.root, model)))
end

"""
    solve_classical(model; starts, refiner=NoRefiner(), execution=SerialMultistart(), time_limit_seconds=nothing)

Own and validate the entire ordered collection of original-coordinate vectors
before refinement. Each input, including a duplicate, has its own classical
start ID and result record. This path needs neither an encoding nor Python.
"""
function solve_classical(model::QHDProblem; starts, refiner::AbstractRefiner=NoRefiner(),
                         execution::MultistartExecution=SerialMultistart(), time_limit_seconds=nothing)
    clock = SolveClock(time_limit_seconds)
    starts isa Union{AbstractVector,Tuple} && !isempty(starts) ||
        throw(ArgumentError("starts must be a nonempty ordered collection of original-coordinate vectors"))
    _check_time(clock, :input_validation)
    owned = _snapshot(model)
    supplied = SuppliedStart[]
    for (ordinal, point) in enumerate(starts)
        _check_time(clock, :input_validation)
        push!(supplied, _supplied_start(owned, point, ordinal))
    end
    structural = !(refiner isa NoRefiner) && _structural_classical(owned)
    return _native_errors() do
        info = _refiner_info(refiner, owned)
        provenance = (; model_fingerprint=_fingerprint(owned, _compiler_schema(owned, :unary)),
                      lower=copy(owned.lower), upper=copy(owned.upper), start_source=:supplied,
                      refiner=info.solver, refiner_details=info, time_limit_seconds=clock.limit)
        phases = (; compilation_seconds=0.0, preparation_seconds=_elapsed(clock.started), decoding_seconds=0.0)
        _finish_result(owned, nothing, TrialRecord[], DecodedStart[], supplied, 0,
                       provenance, refiner, execution, clock, phases; structural)
    end
end

function _candidate(r, stage)
    stage == :selected && return r.selected
    stage in (:initial, :coarse, :refined) || throw(ArgumentError("unknown solution stage $stage"))
    starts = isempty(r.supplied_starts) ? r.sampled_starts : r.supplied_starts
    return _best_stage(starts, r.refinement_records, stage)
end

"""Return an owned best point, a named coordinate, or ordered named coordinates at a stage."""
function solution(r::QHDResult; stage=:selected)
    candidate = _candidate(r, stage)
    return isnothing(candidate) ? nothing : copy(candidate.point)
end

function _name_index(r, name)
    i = findfirst(==(Symbol(name)), r.names)
    isnothing(i) && throw(ArgumentError("unknown variable $name"))
    return i
end

function solution(r::QHDResult, name::Union{Symbol,AbstractString}; stage=:selected)
    i = _name_index(r, name)
    candidate = _candidate(r, stage)
    return isnothing(candidate) ? nothing : candidate.point[i]
end

function solution(r::QHDResult, names::AbstractVector; stage=:selected)
    indices = [_name_index(r, name) for name in names]
    candidate = _candidate(r, stage)
    return isnothing(candidate) ? nothing : candidate.point[indices]
end

"""
    success_probability(result; reference=nothing, epsilon=1e-3, stage=:selected, population=:sampler_trials)

Fraction of all returned trials within strict absolute tolerance of the reference.
With `population=:classical_starts`, count each supplied ordinal once.
Omitting the reference measures agreement with the observed selected minimum.
Returns `missing` for zero trials or an unavailable implicit reference.
"""
function success_probability(r::QHDResult; reference::Union{Nothing,Real}=nothing,
                             epsilon::Real=1e-3, stage=:selected, population=:sampler_trials)
    population in (:sampler_trials, :classical_starts) || throw(ArgumentError("unknown success population $population"))
    starts = _population(r, population, stage)
    isfinite(epsilon) && epsilon > 0 || throw(ArgumentError("epsilon must be finite and positive"))
    isnothing(reference) || isfinite(reference) || throw(ArgumentError("reference must be finite"))
    denominator = population == :sampler_trials ? r.counts.actual_trials : r.counts.classical_requested
    denominator == 0 && return missing
    if isnothing(reference)
        isnothing(r.selected) && return missing
        reference = r.selected.value
    end
    hits = 0
    for st in starts
        candidate = _stage_candidate(st, r.refinement_records[r.attempt_map[st.start_id]], stage)
        if !isnothing(candidate) && abs(candidate.value - reference) < epsilon
            hits += _weight(st)
        end
    end
    return hits / denominator
end

_weight(st::DecodedStart) = st.multiplicity
_weight(::SuppliedStart) = 1

function _population(r, population, stage)
    if population == :classical_starts
        stage in (:initial, :refined, :selected) || throw(ArgumentError("invalid classical stage $stage"))
        return r.supplied_starts
    end
    stage in (:coarse, :selected, :refined) || throw(ArgumentError("invalid sampled stage $stage"))
    return r.sampled_starts
end

"""Mean and lower empirical median over available stage values, weighted by trial or supplied ordinal."""
function objective_summary(r::QHDResult; stage=:coarse, population=:valid_trials)
    population in (:valid_trials, :classical_starts) || throw(ArgumentError("unknown summary population $population"))
    observations = Tuple{Float64,Int}[]
    for st in _population(r, population, stage)
        candidate = _stage_candidate(st, r.refinement_records[r.attempt_map[st.start_id]], stage)
        isnothing(candidate) || push!(observations, (candidate.value, _weight(st)))
    end
    weight = sum(last, observations; init=0)
    weight == 0 && return (; population, stage, weight, mean=missing, median=missing)
    sort!(observations; by=first)
    cumulative = 0
    median_value = first(observations)[1]
    for (value, n) in observations
        cumulative += n
        if cumulative >= cld(weight, 2)
            median_value = value
            break
        end
    end
    weighted_sum = sum(_exact(value) * n for (value, n) in observations)
    return (; population, stage, weight, mean=Float64(weighted_sum / weight), median=median_value)
end

# One Cartesian grid per original weighted term, in original variable order.
# The coordinates are exactly the Float64 values reconstructed by the decoder.
function _term_axis(model, i, r, levels, checkpoint)
    lo, hi = _exact(model.lower[i]), _exact(model.upper[i])
    return [Float64(lo + (hi-lo)*(_exact(k)/r)) for k in _checked_grid(levels, checkpoint)]
end

function _term_context(f, label)
    try
        return f()
    catch err
        err isa Union{ArgumentError,DomainError,OverflowError} || rethrow()
        _numerical_admission(ArgumentError("$label: $(sprint(showerror, err))"))
    end
end

function _exp_centering(term, anchor, label, checkpoint)
    # Anchor exponent arguments only after the affine box substitution. Add
    # the two shifts as exact dyadics before combining them with log|alpha|
    # at high precision. Never form a potentially unrepresentable exp(shift).
    shifts = [_coefficient(_evaluate(factor, anchor, checkpoint), "$label exponent anchor") for factor in term.factors]
    alpha = term.weight
    iszero(alpha) && throw(ArgumentError("$label: a zero exponential weight must be simplified first"))
    magnitude = _with_reference_precision(256) do
        exp(log(BigFloat(abs(alpha))) + BigFloat(sum(_exact, shifts)))
    end
    gamma = copysign(Float64(magnitude), alpha)
    isfinite(gamma) && !iszero(gamma) ||
        throw(ArgumentError("$label: centered exponential multiplier is zero or nonfinite in Float64 at the grid anchor"))
    return shifts, gamma
end

function _term_factor(term, a, axis, anchor, shifts, label, encoding, grid, checkpoint)
    coordinate = term.variables[a]
    point = copy(anchor)
    values = map(_checked_grid(eachindex(axis), checkpoint)) do k
        point[coordinate] = axis[k]
        _term_context("$label, Float64 factor $a, coordinate $coordinate at $(axis[k])") do
            # After fixed substitution there may be just one free factor.
            # Tabulate its complete weighted value before any Float64 cast,
            # e.g. 1e-300 * exp(x+200) at x=600.
            value = if term.kind == :univariate
                _with_reference_precision(256) do
                    BigFloat(term.weight) * _evaluate(term.factors[a], BigFloat.(point), checkpoint)
                end
            else
                _evaluate(term.factors[a], point, checkpoint)
            end
            if term.kind == :exp_sum
                value = exp(Float64(_exact(value) - _exact(shifts[a])))
                isfinite(value) && value > 0 ||
                    throw(ArgumentError("centered exponential factor is zero or nonfinite"))
            elseif iszero(value)
                # A zero of the actual function is legitimate; a lost nonzero
                # factor must not be mistaken for a structural zero.
                zero_check = term.kind == :univariate ? term.reference : term.factors[a]
                ref = _reference(zero_check, point, checkpoint)
                iszero(ref.value) || throw(ArgumentError("nonzero factor underflowed to zero"))
            end
            _coefficient(value, "factor value")
        end
    end
    return _table_lift(encoding, term.kind, coordinate, grid,
                       map(_exact, _checked_grid(values, checkpoint)), checkpoint)
end

# Ranges enclose independent evaluations of the original expression tree.
# Keep MPFR / exact endpoints temporary. The digest and public metadata use
# outward-rounded Float64 bounds, not architecture-dependent MPFR internals.
_term_key(levels, index) = ntuple(a -> levels[a][index[a]], length(levels))

function _term_references(term, axes, anchor, levels, label, checkpoint)
    point = copy(anchor)
    bits = 0
    ranges = map(_checked_grid(CartesianIndices(Tuple(length.(axes))), checkpoint)) do index
        for a in eachindex(axes)
            point[term.variables[a]] = axes[a][index[a]]
        end
        key = _term_key(levels, index)
        reference = _term_context("$label at levels $key, original weighted reference") do
            ref = _reference(term.reference, point, checkpoint)
            _coefficient(ref.value, "original weighted value at $key")
            ref
        end
        bits = max(bits, reference.precision)
        (_exact(reference.lower), _exact(reference.upper))
    end
    return ranges, bits
end

# Evaluate the stored local lift on each valid codeword, in the table's own
# grid order. Unary adds trailing slots; a one-hot block selects one slot.
function _lifted_values(::Unary, f, checkpoint)
    values = Vector{Rational{BigInt}}(undef, length(f.coefficients) + 1)
    values[1] = _exact(f.baseline)
    for (i, c) in _checked_grid(enumerate(Iterators.reverse(f.coefficients)), checkpoint)
        values[i+1] = values[i] + _exact(c)
    end
    return values
end

function _lifted_values(::OneHot, f, checkpoint)
    baseline = _exact(f.baseline)
    return [baseline + _exact(c) for c in _checked_grid(f.coefficients, checkpoint)]
end

function _structured_tables(term, ordinal, model, transform, plan, checkpoint)
    label = "structured term $ordinal ($(term.kind), variables $(term.variables))"
    variables = term.variables
    blocks = [searchsortedfirst(transform.free, i) for i in variables]
    resolutions = plan.resolutions[blocks]
    levels = [_table_levels(plan.spec, r) for r in resolutions]
    grids = [[Float64(_exact(k)/r) for k in _checked_grid(ks, checkpoint)] for (ks,r) in zip(levels,resolutions)]
    axes = [_term_axis(model, i, r, ks, checkpoint) for (i,r,ks) in zip(variables,resolutions,levels)]
    # Anchor at the lowest reachable levels. Source one-hot omits the lower
    # box endpoint, so this need not be the lower bound used by unary.
    anchor = copy(model.lower)
    for (i, axis) in zip(variables, axes)
        anchor[i] = minimum(axis)
    end
    shifts, weight = if term.kind == :exp_sum
        _term_context("$label, Float64 centering") do
            _exp_centering(term, anchor, label, checkpoint)
        end
    elseif term.kind == :univariate
        Float64[], 1.0  # its table contains the weighted values
    else
        Float64[], term.weight
    end
    factors = [_term_factor(term, a, axes[a], anchor, shifts, label, plan.spec, grids[a], checkpoint)
               for a in eachindex(axes)]
    pieces = if term.kind == :log_product
        [(; blocks=[b], factors=[f], weight) for (b,f) in zip(blocks,factors)]
    else
        [(; blocks, factors, weight)]
    end

    ranges, bits = _term_references(term, axes, anchor, levels, label, checkpoint)
    checked_ranges = _checked_grid(ranges, checkpoint)
    reference_min = minimum(first, checked_ranges)
    reference_max = maximum(last, checked_ranges)
    # Lower bound on the true maximum magnitude makes the normalized test
    # conservative even when an interval encloses an irrational reference.
    scale = maximum(interval -> max(0, first(interval), -last(interval)), checked_ranges)
    if iszero(scale) && (term.kind == :exp_sum || !iszero(reference_min) || !iszero(reference_max))
        _numerical_admission(ArgumentError("$label: unresolved or zero reference scale on the original weighted grid"))
    end
    exact_weight = _exact(weight)
    # Exact arithmetic evaluates the expanded local c,d,B used by the global
    # accumulator before its final Float64 rounding.
    lifted = [_lifted_values(plan.spec, f, checkpoint) for f in factors]
    table_values = [map(_exact, _checked_grid(f.values, checkpoint)) for f in factors]
    combine = term.kind == :log_product ? sum : prod
    lowered_error = lifted_error = zero(Rational{BigInt})
    for index in _checked_grid(CartesianIndices(ranges), checkpoint)
        key = _term_key(levels, index)
        raw = exact_weight * combine(table_values[a][index[a]] for a in eachindex(factors))
        lowered = _exact(_compiled_coefficient(raw, "$label, lowered Float64 value at levels $key"))
        energy = exact_weight * combine(lifted[a][index[a]] for a in eachindex(factors))
        lo, hi = ranges[index]
        for (stage, candidate) in ((:lowered, lowered), (:lifted, energy))
            # Check the complete uncertainty interval, under both required
            # inequalities. A normalized absolute tolerance uses scale > 0.
            error = max(abs(candidate-lo), abs(candidate-hi))
            bound = _GRID_RTOL * max(abs(candidate), lo, -hi)
            error <= _GRID_ATOL + bound && (iszero(scale) || error <= _GRID_ATOL*scale + bound) ||
                _numerical_admission(ArgumentError("$label, levels $key: $stage Float64 energy fails original weighted raw/normalized admission"))
            if stage == :lowered
                lowered_error = max(lowered_error, error)
            else
                lifted_error = max(lifted_error, error)
            end
        end
    end
    normalized(error) = iszero(scale) ? 0.0 : Float64(error / scale, RoundUp)
    lower_bound, upper_bound = Float64(reference_min, RoundDown), Float64(reference_max, RoundUp)
    isfinite(lower_bound) && isfinite(upper_bound) ||
        _numerical_admission(ArgumentError("$label: original weighted grid range exceeds finite Float64 bounds"))
    handling = term.kind == :exp_sum ? :signed_logscale :
               term.kind == :univariate ? :weighted_table : :external
    provenance = (; rule=term.kind == :exp_sum ? :centered_exp : term.kind,
                  ordinal, variables=copy(variables), shifts, multiplier=term.weight,
                  effective_multiplier=weight, multiplier_handling=handling, reference_bits=bits,
                  reference_min=lower_bound, reference_max=upper_bound,
                  atol=Float64(_GRID_ATOL), rtol=Float64(_GRID_RTOL),
                  lowered_max_raw_residual=Float64(lowered_error, RoundUp),
                  lowered_max_normalized_residual=normalized(lowered_error),
                  lifted_max_raw_residual=Float64(lifted_error, RoundUp),
                  lifted_max_normalized_residual=normalized(lifted_error),
                  max_lifted_error=Float64(lifted_error, RoundUp))
    if plan.spec isa OneHot
        provenance = merge(provenance, (; grid=:source_onehot_descending, baseline_policy=:zero,
                                         anchor_coordinates=anchor[variables]))
    end
    return (; pieces, provenance)
end

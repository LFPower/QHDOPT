# Source one-hot uses descending positive grid levels.
# Obtain conservative scale bounds from valid assignments.
function _onehot_witness_scale(base, plan, checkpoint=nothing)
    r = plan.spec.r
    reference(i) = i - mod(i-1, r)
    fields = _exact.(base.linear_bits)
    anchor = _exact(base.energy_offset) + sum(offset -> fields[offset+1], plan.offsets)
    scale = zero(anchor)
    B = base.coupling_bits
    for (i, j, value) in _checked_grid(_upper_entries(B), checkpoint)
        a, b = reference(i), reference(j)
        h = _exact(value)
        if i == a
            fields[j] += h
        end
        if j == b
            fields[i] += h
        end
        if i == a && j == b
            anchor += h
        elseif i != a && j != b
            mixed = h - _exact(B[a,j]) - _exact(B[i,b]) + _exact(B[a,b])
            scale = max(scale, abs(mixed)/4)
        end
    end
    for i in _checked_grid(eachindex(fields), checkpoint)
        scale = max(scale, abs(fields[i] - fields[reference(i)])/2)
    end
    return max(scale, abs(anchor))
end

function _table_lift(::OneHot, term, coordinate, grid, exact_values, checkpoint=nothing)
    label = "$term one-hot table at coordinate $coordinate"
    scale = maximum(abs, _checked_grid(exact_values, checkpoint))
    values = Vector{Float64}(undef, length(grid))
    raw, normalized = 0.0, 0.0
    for slot in _checked_grid(eachindex(grid), checkpoint)
        value = _compiled_coefficient(exact_values[slot], "$label slot $slot")
        values[slot] = value
        # Exactly one hot slot selects exactly this coefficient.
        a, b = _grid_check(exact_values[slot], _exact(value), scale, "$label slot $slot")
        raw, normalized = max(raw, a), max(normalized, b)
    end
    return EncodingTable(term, coordinate, copy(grid), values, 0.0, copy(values), raw, normalized)
end

function _onehot_qp_base(model, normalized, transform, plan, tables, storage)
    # Every valid block selects a diagonal/linear table value. Cross-variable
    # terms select one entry of the outer product of the two identity lifts.
    linear = reduce(vcat, (table.coefficients for table in @view tables[2:end]))
    slot_values = _exact.(first(tables).coefficients)
    widths = _exact.(model.upper[transform.free]) - _exact.(model.lower[transform.free])
    rows, cols, entries = Int[], Int[], Float64[]
    for (i, j, _) in _upper_entries(normalized.H)
        i == j && continue
        a, b = transform.free[i], transform.free[j]
        pair = _exact(model.objective.H[a,b]) * widths[i] * widths[j]
        for v in eachindex(slot_values), u in eachindex(slot_values)
            coupling = _compiled_coefficient(pair * slot_values[u] * slot_values[v], "one-hot coupling ($a:$u, $b:$v)")
            row, col = plan.offsets[i]+u, plan.offsets[j]+v
            push!(rows, row, col)
            push!(cols, col, row)
            push!(entries, coupling, coupling)
        end
    end
    B = _stored(sparse(rows, cols, entries, plan.n_bits, plan.n_bits), storage)
    return CanonicalBinaryEnergy(normalized.constant, linear, B)
end

# Check each selected table value and cross-block coefficient against the QP
# evaluated at the *returned* Float64 coordinates, including interpolation
# roundoff. The sum of the worst local residuals bounds all valid codewords
# without a Cartesian product over all variables.
function _onehot_qp_fidelity(model, transform, normalized, exact_linear, plan, base, spin_rounding)
    lower = _exact_value(model.objective, _exact.(model.lower))
    upper = lower
    error_bound = abs(lower - _exact(base.energy_offset)) + spin_rounding
    widths = _exact.(model.upper[transform.free]) - _exact.(model.lower[transform.free])
    coordinates = Vector{Vector{Rational{BigInt}}}(undef, length(widths))
    for (i, a) in enumerate(transform.free)
        lo, width, r = _exact(model.lower[a]), widths[i], plan.resolutions[i]
        # Keep the coordinates in precisely the same slot order as the lift.
        coordinates[i] = [(_exact(Float64(lo + width*k/r))-lo)/width for k in _table_levels(plan.spec, r)]
        quadratic = _exact(model.objective.H[a,a]) * width^2 / 2
        values = [t * (exact_linear[i] + quadratic*t) for t in coordinates[i]]
        lower += minimum(values)
        upper += maximum(values)
        scale, local_error = maximum(abs, values), zero(error_bound)
        for (u, reference) in enumerate(values)
            encoded = _exact(base.linear_bits[plan.offsets[i]+u])
            _grid_check(reference, encoded, scale, "one-hot coordinate $a slot $u")
            local_error = max(local_error, abs(encoded - reference))
        end
        error_bound += local_error
    end
    for (i, j, _) in _upper_entries(normalized.H)
        i == j && continue
        a, b = transform.free[i], transform.free[j]
        pair = _exact(model.objective.H[a,b]) * widths[i] * widths[j]
        # The reachable grid includes one and excludes zero; all returned
        # normalized coordinates are nonnegative and ordered from high to low.
        smallest_product = pair * last(coordinates[i]) * last(coordinates[j])
        lower += min(pair, smallest_product)
        upper += max(pair, smallest_product)
        local_error = zero(error_bound)
        for (v, t) in enumerate(coordinates[j]), (u, s) in enumerate(coordinates[i])
            reference = pair * s * t
            encoded = _exact(base.coupling_bits[plan.offsets[i]+u, plan.offsets[j]+v])
            _grid_check(reference, encoded, abs(pair), "one-hot pair ($a:$u, $b:$v)")
            local_error = max(local_error, abs(encoded - reference))
        end
        error_bound += local_error
    end
    min_abs = max(0, lower, -upper)
    scale_floor = max(min_abs, _onehot_witness_scale(base, plan) - error_bound)
    certified = error_bound <= _GRID_ATOL + _GRID_RTOL*min_abs &&
                error_bound <= _GRID_ATOL*scale_floor + _GRID_RTOL*min_abs
    return certified, Float64(error_bound, RoundUp)
end

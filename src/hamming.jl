# Build exchangeable binary fields and couplings for the bounded quadratic.
function _hamming_binary(model, normalized, transform, plan, exact_linear, storage)
    free = transform.free
    widths = _exact.(model.upper[free]) - _exact.(model.lower[free])
    steps = widths ./ plan.resolutions
    linear_bits = Vector{Float64}(undef, plan.n_bits)
    for (i, original) in enumerate(free)
        r = plan.resolutions[i]
        h = _exact(model.objective.H[original, original])
        field = exact_linear[i]/r + h*steps[i]^2/2
        label = "Hamming field for coordinate $original"
        stored = _compiled_coefficient(field, label)
        _grid_check(field, _exact(stored), abs(field), label)
        offset = plan.offsets[i]
        fill!(@view(linear_bits[offset+1:offset+r]), stored)
    end

    rows, cols, entries = Int[], Int[], Float64[]
    for (i, j, _) in _upper_entries(normalized.H)
        ri, rj = plan.resolutions[i], plan.resolutions[j]
        i == j && ri == 1 && continue  # no distinct pair within this block
        a, b = free[i], free[j]
        coupling = _exact(model.objective.H[a,b]) * steps[i] * steps[j]
        label = "Hamming coupling for coordinates ($a,$b)"
        stored = _compiled_coefficient(coupling, label)
        _grid_check(coupling, _exact(stored), abs(coupling), label)
        for slot_j in 1:rj
            last_i = i == j ? slot_j-1 : ri
            v = plan.offsets[j] + slot_j
            for slot_i in 1:last_i
                u = plan.offsets[i] + slot_i
                push!(rows, u, v)
                push!(cols, v, u)
                push!(entries, stored, stored)
            end
        end
    end
    B = _stored(sparse(rows, cols, entries, plan.n_bits, plan.n_bits), storage)
    return CanonicalBinaryEnergy(normalized.constant, linear_bits, B)
end

# The kth active bit interacts with each of the k-1 previously active bits.
# Exchangeability makes this increment identical for every codeword of weight k.
function _hamming_increment(base, offset, k)
    first_slot = offset+1
    field = _exact(base.linear_bits[first_slot])
    k == 1 && return field
    return field + (k-1)*_exact(base.coupling_bits[first_slot,first_slot+1])
end

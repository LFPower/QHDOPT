"""The retained decomposition and affine coordinate map for a structured objective."""
struct NormalizedStructuredObjective
    original::StructuredObjective
    transform::BoxTransform
    constant::Float64
    terms::Vector{StructuredTerm}
end

objective_value(f::NormalizedStructuredObjective, z) = objective_value(f.original, reconstruct(f.transform, z))
_compiled_value(obj::StructuredObjective, x, checkpoint) = Float64(_compiled_reference(obj.root, x, checkpoint).value)
_grid_reference(obj::StructuredObjective, x, checkpoint) = _exact(_compiled_value(obj, x, checkpoint))

# Accumulate repeated lifted terms exactly before converting coefficients.
function _structured_binary(constant, tables, plan, storage, checkpoint)
    offset = _exact(constant)
    linear = zeros(Rational{BigInt}, plan.n_bits)
    upper = Dict{Tuple{Int,Int},Rational{BigInt}}()
    for record in tables, piece in record.pieces
        weight = _exact(piece.weight)
        left = piece.factors[1]
        i = plan.offsets[piece.blocks[1]]
        if length(piece.factors) == 1
            offset += weight * _exact(left.baseline)
            for (u, a) in _checked_grid(enumerate(left.coefficients), checkpoint)
                linear[i+u] += weight * _exact(a)
            end
            continue
        end
        right = piece.factors[2]
        j = plan.offsets[piece.blocks[2]]
        i != j || error("pair factors must belong to distinct encoding blocks")
        beta, gamma = _exact(left.baseline), _exact(right.baseline)
        offset += weight * beta * gamma
        for (v, b) in _checked_grid(enumerate(right.coefficients), checkpoint)
            linear[j+v] += weight * beta * _exact(b)
        end
        for (u, a) in _checked_grid(enumerate(left.coefficients), checkpoint)
            wa = weight * _exact(a)
            linear[i+u] += wa * gamma
            iszero(wa) && continue
            for (v, b) in _checked_grid(enumerate(right.coefficients), checkpoint)
                iszero(b) && continue
                key = minmax(i+u, j+v)
                upper[key] = get(upper, key, 0) + wa * _exact(b)
            end
        end
    end
    c = _compiled_coefficient(offset, "assembled structured offset")
    d = [_compiled_coefficient(a, "assembled structured slot $i")
         for (i, a) in _checked_grid(enumerate(linear), checkpoint)]
    rounding = abs(offset - _exact(c)) +
               sum(i -> abs(linear[i] - _exact(d[i])), _checked_grid(eachindex(d), checkpoint))
    rows, cols, entries = Int[], Int[], Float64[]
    # The upper-triangle keys are unique. sparse orders the final CSC entries.
    for ((i, j), value) in _checked_grid(upper, checkpoint)
        iszero(value) && continue
        coupling = _compiled_coefficient(value, "assembled coupling ($i,$j)")
        rounding += abs(value - _exact(coupling))
        push!(rows, i, j)
        push!(cols, j, i)
        push!(entries, coupling, coupling)
    end
    B = _stored(sparse(rows, cols, entries, plan.n_bits, plan.n_bits), storage)
    return CanonicalBinaryEnergy(c, d, B), rounding
end

# Bound the complete objective on its grid using the reference ranges and
# lifted-energy discrepancies recorded at each term's LOCAL grid. Cancellation
# can make relative agreement harder to certify even when every term passes.
function _structured_fidelity(decomposition, tables, base, rounding, plan, checkpoint)
    constant_error = _exact(decomposition.constant_error)
    lower = _exact(decomposition.constant) - constant_error
    upper = _exact(decomposition.constant) + constant_error
    error_bound = rounding + constant_error
    for record in _checked_grid(tables, checkpoint)
        p = record.provenance
        lower += _exact(p.reference_min)
        upper += _exact(p.reference_max)
        error_bound += _exact(p.max_lifted_error)
    end
    encoded_scale = if plan.spec isa OneHot
        _onehot_witness_scale(base, plan, checkpoint)
    else
        # For unary, the all-zero anchor and adjacent one- and two-block
        # differences provide lower bounds on the maximum absolute energy.
        scale = max(abs(_exact(base.energy_offset)), _exact(maximum(abs, base.linear_bits))/2)
        for (_, _, b) in _checked_grid(_upper_entries(base.coupling_bits), checkpoint)
            scale = max(scale, abs(_exact(b))/4)
        end
        scale
    end
    min_abs = max(0, lower, -upper)
    scale_floor = max(min_abs, encoded_scale - error_bound)
    certified = max(abs(lower), abs(upper)) <= _exact(floatmax(Float64)) &&
                error_bound <= _GRID_ATOL + _GRID_RTOL*min_abs &&
                error_bound <= _GRID_ATOL*scale_floor + _GRID_RTOL*min_abs
    return certified, error_bound
end

"""
    compile(model::QHDProblem{StructuredObjective}; encoding, validity, storage=:sparse)

Substitute fixed coordinates, admit the ordered decomposition on its local
grids, and assemble its unary or source one-hot energy. Each entry of `prepared.tables` contains
the factor tables and the original weighted term's numerical audit.
"""
function compile(model::QHDProblem{StructuredObjective}; encoding::GridEncoding, validity=_default_validity(encoding),
                 storage::Symbol=:sparse)
    return _native_errors(() -> _compile(model, encoding, validity, storage, () -> nothing))
end

function _compile(model::QHDProblem{StructuredObjective}, encoding, validity, storage, checkpoint)
    encoding isa Hamming && _invalid_option(:configuration, ArgumentError("Hamming compilation is certified only for bounded QPs; use quadratic_problem(H, b; ...)"))
    storage in (:sparse, :dense) || _invalid_option(:configuration, ArgumentError("storage must be :sparse or :dense"))
    snapshot = _snapshot(model, checkpoint)
    transform = BoxTransform(snapshot)
    plan = _encoding_plan(encoding, validity, transform)
    decomposition = _decompose(snapshot, checkpoint)
    checkpoint()
    normalized = NormalizedStructuredObjective(snapshot.objective, transform, decomposition.constant,
                                               decomposition.terms)
    shortcut = if isempty(transform.free)
        :all_fixed
    elseif isempty(decomposition.terms)
        :constant
    else
        :none
    end
    if shortcut == :none
        isnothing(transform.widths) && _numerical_admission(ArgumentError("nonfinite free-coordinate width"))
        _check_grid_coordinates(transform, (_exact(k)/encoding.r for k in _table_levels(encoding, encoding.r)), checkpoint)
    end
    tables = [_structured_tables(term, i, snapshot, transform, plan, checkpoint)
              for (i, term) in enumerate(decomposition.terms)]
    checkpoint()
    if shortcut == :none
        base, assembly_error = _structured_binary(normalized.constant, tables, plan, storage, checkpoint)
        binary, penalty_error = _apply_validity(base, plan, checkpoint)
        problem, spin_offset, spin_error = _sampling_data(base, binary, plan, storage, penalty_error, checkpoint)
        checkpoint()
        certified, bound = _structured_fidelity(decomposition, tables, base, assembly_error + spin_error, plan, checkpoint)
        certified || _check_small_grid(snapshot, transform, plan, base, binary, problem, spin_offset, checkpoint)
        check = certified ? :interval_certificate : :enumerated_grid
    else
        base = binary = problem = nothing
        spin_offset = _compiled_value(snapshot.objective, _shortcut_point(transform, plan), checkpoint)
        assembly_error = bound = 0
        check = :deterministic
    end
    diagnostics = (policy=Symbol("structured_", _encoding_id(encoding), "_v1"), grid=_grid_convention(encoding),
                   standalone_constants=:explicit, grid_atol=Float64(_GRID_ATOL), grid_rtol=Float64(_GRID_RTOL),
                   whole_objective_check=check, grid_error_upper_bound=Float64(bound, RoundUp),
                   assembly_error_bound=Float64(assembly_error, RoundUp), constant_error_bound=decomposition.constant_error,
                   original_dimension=length(snapshot.names), free_dimension=length(transform.free),
                   bit_dimension=plan.n_bits, spin_nnz=isnothing(problem) ? 0 : count(!iszero, problem.Q))
    return _artifact(snapshot, transform, normalized, plan, tables, base, binary, problem,
                     spin_offset, shortcut, storage, diagnostics)
end

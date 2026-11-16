struct EncodingTable
    term::Symbol
    coordinate::Int               # original coordinate, or 0 for the common z factor
    grid::Vector{Float64}
    values::Vector{Float64}
    baseline::Float64
    coefficients::Vector{Float64} # in bit slot order
    max_raw_residual::Float64
    max_normalized_residual::Float64
end
# Existing code may refer to the table type when inspecting unary artifacts.
const UnaryTable = EncodingTable

struct CanonicalBinaryEnergy
    energy_offset::Float64
    linear_bits::Vector{Float64}
    coupling_bits::CoefficientMatrix
end

struct CompiledQHD{M<:QHDProblem,N,T}
    model::M
    model_fingerprint::String
    transform::BoxTransform
    normalized::N
    encoding::EncodingPlan
    tables::Vector{T}
    base_energy::Union{Nothing,CanonicalBinaryEnergy}
    binary_energy::Union{Nothing,CanonicalBinaryEnergy}
    sampling_problem::Union{Nothing,QuadraticProblem}
    spin_offset::Float64
    shortcut::Symbol
    storage::Symbol
    diagnostics::NamedTuple
    fingerprint::String
end

# Versioned, length-delimited serialization in schema field order. Include CSC
# storage (including indices), dimensions, and the IEEE bit patterns of values.
# It is independent of Julia's process-local hash seed and host byte order.
_digest_part(io, x::Int) = write(io, htol(Int64(x)))
_digest_part(io, x::Float64) = write(io, htol(reinterpret(UInt64, x)))
function _digest_part(io, x::Union{String,Symbol})
    bytes = codeunits(String(x))
    _digest_part(io, length(bytes))
    write(io, bytes)
end
function _digest_part(io, x::AbstractArray)
    _digest_part(io, :array)
    _digest_part(io, ndims(x))
    foreach(n -> _digest_part(io, n), size(x))
    foreach(v -> _digest_part(io, v), x)
end
function _digest_part(io, x::Array{T}) where {T<:Union{Int64,Float64}}
    _digest_part(io, :array64)
    _digest_part(io, ndims(x))
    foreach(n -> _digest_part(io, n), size(x))
    words = reinterpret(UInt64, vec(x))
    bytes = reinterpret(UInt8, Base.ENDIAN_BOM == 0x04030201 ? words : htol.(words))
    # Hash numeric leaves directly, so a second complete set of coefficient
    # arrays need not reside in the serialization buffer.
    write(io, sha256(bytes))
end
function _digest_part(io, x::SparseMatrixCSC)
    _digest_part(io, :csc)
    foreach(n -> _digest_part(io, n), size(x))
    foreach(v -> _digest_part(io, v), (x.colptr, rowvals(x), nonzeros(x)))
end
function _digest_record(io, x, tag)
    _digest_part(io, tag)
    _digest_part(io, fieldcount(typeof(x)))
    for field in fieldnames(typeof(x))
        _digest_part(io, getfield(x, field))
    end
end
_digest_part(io, x) = _digest_record(io, x, nameof(typeof(x)))
# Keep the v1 table wire tag. The encoding spec and artifact schema identify
# the lifting policy; a shared in-memory record does not invalidate unary IDs.
_digest_part(io, x::EncodingTable) = _digest_record(io, x, :UnaryTable)
_model_kind(::QuadraticObjective) = :qp
_model_kind(::StructuredObjective) = :structured
_compiler_schema(model, encoding_id) = "QHDOPT/$(_model_kind(model.objective))-$encoding_id/v1"

function _fingerprint(data, schema)
    io = IOBuffer()
    _digest_part(io, schema)
    _digest_part(io, data)
    return bytes2hex(sha256(take!(io)))
end

function _validate_artifact(prepared::CompiledQHD)
    payload = ntuple(i -> getfield(prepared, i), fieldcount(typeof(prepared)) - 1)
    schema = _compiler_schema(prepared.model, _encoding_id(prepared.encoding.spec))
    _fingerprint(payload, schema) == prepared.fingerprint ||
        throw(ArgumentError("compiled artifact has been modified; recompile its model"))
    return nothing
end

"""Inspect all coefficients, tables, and provenance as an independent (potentially large) copy."""
function inspect(prepared::CompiledQHD)
    _validate_artifact(prepared)
    return deepcopy(prepared)
end

const _GRID_ATOL = 1 // BigInt(10)^12
const _GRID_RTOL = 1 // BigInt(10)^10

# Compare in exact arithmetic to avoid overflow in subtraction and to keep
# the scale-normalized check meaningful for very small nonzero objectives.
function _grid_check(reference, actual, scale, label)
    residual = abs(reference - actual)
    relative = iszero(scale) ? residual : residual / scale
    bound = _GRID_RTOL * max(abs(reference), abs(actual))
    residual <= _GRID_ATOL + bound && residual <= _GRID_ATOL * scale + bound ||
        _numerical_admission(ArgumentError("$label failed raw/normalized energy admission"))
    return Float64(residual), Float64(relative)
end

# Keep the shape of grid iterators (including CartesianIndices) while bounding
# the number of points processed between cooperative clock checks.
function _checked_grid(items, checkpoint)
    return (begin
        (k - 1) % 32 == 0 && checkpoint()
        item
    end for (k, item) in enumerate(items))
end
_checked_grid(items, ::Nothing) = items

function _unary_table(term, coordinate, grid, exact_values, checkpoint=nothing)
    label = "$term table at coordinate $coordinate"
    values = [_compiled_coefficient(v, "$label level $(k-1)")
              for (k, v) in _checked_grid(enumerate(exact_values), checkpoint)]
    increments = [_compiled_coefficient(exact_values[k+1] - exact_values[k], "$label finite difference")
                  for k in _checked_grid(1:length(values)-1, checkpoint)]
    scale = maximum(abs, _checked_grid(exact_values, checkpoint))
    lifted = _exact(values[1])
    raw, normalized = 0.0, 0.0
    for k in _checked_grid(eachindex(values), checkpoint)
        k > 1 && (lifted += _exact(increments[k-1]))
        for evaluation in (_exact(values[k]), lifted)
            a, b = _grid_check(exact_values[k], evaluation, scale, "$label level $(k-1)")
            raw, normalized = max(raw, a), max(normalized, b)
        end
    end
    return EncodingTable(term, coordinate, copy(grid), values, values[1], reverse(increments),
                      raw, normalized)
end

_table_lift(::Unary, args...) = _unary_table(args...)

function _check_grid_coordinates(transform, levels, checkpoint=nothing)
    for i in transform.free
        lo = _exact(transform.lower[i])
        width = _exact(transform.upper[i]) - lo
        for t in _checked_grid(levels, checkpoint)
            x = Float64(lo + width*t)
            _grid_check(t, (_exact(x)-lo)/width, one(t), "original coordinate $i at normalized grid point $t")
        end
    end
end

function _qp_tables(model, transform, encoding, exact_linear)
    r = encoding.r
    levels = [_exact(k) / r for k in _table_levels(encoding, r)]
    _check_grid_coordinates(transform, levels)
    grid = Float64.(levels)
    # Every normalized quadratic cross term uses the same identity factor.
    tables = [_table_lift(encoding, :identity, 0, grid, levels)]
    for (j, i) in enumerate(transform.free)
        lo, hi = _exact(model.lower[i]), _exact(model.upper[i])
        width = hi - lo
        diagonal = _exact(model.objective.H[i,i]) * width^2 / 2
        values = [diagonal * t^2 + exact_linear[j] * t for t in levels]
        push!(tables, _table_lift(encoding, :diagonal_linear, i, grid, values))
    end
    return tables
end

_stored(matrix, storage) = storage == :dense ? Matrix(matrix) : matrix

function _binary_energies(model, normalized, transform, plan, tables, storage)
    r, m = plan.spec.r, plan.n_bits
    d = reduce(vcat, (t.coefficients for t in @view tables[2:end]))
    rows, cols, entries = Int[], Int[], Float64[]
    # Each upper-triangular pair has a disjoint rectangular block. There
    # are no same-slot products to fold: diagonal QP terms were table-lifted.
    for (i, j, h) in _upper_entries(normalized.H)
        i == j && continue
        a, b = transform.free[i], transform.free[j]
        widths = (_exact(model.upper[a]) - _exact(model.lower[a])) *
                 (_exact(model.upper[b]) - _exact(model.lower[b]))
        exact_pair = widths * _exact(model.objective.H[a,b])
        coupling = _compiled_coefficient(exact_pair / (BigInt(r)^2), "pair ($a,$b) binary coupling")
        # Every local pair-grid value is k*l times this common coupling.
        # The endpoint check therefore bounds all local residuals exactly.
        _grid_check(exact_pair, _exact(coupling)*BigInt(r)^2, abs(exact_pair), "pair ($a,$b)")
        for v in plan.offsets[j]+1:plan.offsets[j]+r, u in plan.offsets[i]+1:plan.offsets[i]+r
            push!(rows, u, v)
            push!(cols, v, u)
            push!(entries, coupling, coupling)
        end
    end
    B = _stored(sparse(rows, cols, entries, m, m), storage)
    return CanonicalBinaryEnergy(normalized.constant, d, B)
end

function _apply_validity(base, plan, checkpoint=nothing)
    validity = plan.validity
    if !(validity isa FoldedPenalty) || (plan.spec isa Unary && plan.spec.r == 1)
        return base, zero(Rational{BigInt})
    end
    return _fold_penalty(plan.spec, base, plan, _exact(validity.weight), checkpoint)
end

function _fold_penalty(::Unary, base, plan, weight, checkpoint)
    penalized = copy(base.linear_bits)
    rounding = zero(weight)
    rows, cols, entries = Int[], Int[], Float64[]
    for offset in plan.offsets, slot in _checked_grid(1:plan.spec.r-1, checkpoint)
        u = offset + slot
        exact_sum = _exact(base.linear_bits[u]) + weight
        penalized[u] = _compiled_coefficient(exact_sum, "penalized linear coefficient $u")
        rounding += abs(_exact(penalized[u]) - exact_sum)
        push!(rows, u, u+1)
        push!(cols, u+1, u)
        push!(entries, -Float64(weight), -Float64(weight))
    end
    penalized_B = base.coupling_bits + sparse(rows, cols, entries, plan.n_bits, plan.n_bits)
    return CanonicalBinaryEnergy(base.energy_offset, penalized, penalized_B), rounding
end

function _fold_penalty(::OneHot, base, plan, weight, checkpoint)
    # lambda * (sum(q)-1)^2 contributes a constant even for a one-slot block.
    offset = _exact(base.energy_offset) + length(plan.offsets)*weight
    constant = _compiled_coefficient(offset, "one-hot penalty offset")
    rounding = abs(_exact(constant) - offset)
    linear = similar(base.linear_bits)
    for i in _checked_grid(eachindex(linear), checkpoint)
        field = _exact(base.linear_bits[i]) - weight
        linear[i] = _compiled_coefficient(field, "one-hot penalized linear coefficient $i")
        rounding += abs(_exact(linear[i]) - field)
    end
    rows, cols, entries = Int[], Int[], Float64[]
    if plan.spec.r > 1
        coupling = _compiled_coefficient(2*weight, "one-hot penalty coupling")
        for (block, r) in zip(plan.offsets, plan.resolutions), j in 2:r, i in _checked_grid(1:j-1, checkpoint)
            push!(rows, block+i, block+j)
            push!(cols, block+j, block+i)
            push!(entries, coupling, coupling)
        end
    end
    # All base pairs in a table encoding span two distinct blocks.
    B = base.coupling_bits + sparse(rows, cols, entries, plan.n_bits, plan.n_bits)
    return CanonicalBinaryEnergy(constant, linear, B), rounding
end

function _linear_constraints(::Unary, plan, storage)
    rows, cols, entries = Int[], Int[], Float64[]
    count = 0
    for offset in plan.offsets, j in 1:plan.spec.r-1
        count += 1
        push!(rows, count, count)
        push!(cols, offset+j, offset+j+1)
        push!(entries, -1, 1)
    end
    return (; A=_stored(sparse(rows, cols, entries, count, plan.n_bits), storage))
end

function _linear_constraints(::OneHot, plan, storage)
    # q=(1-s)/2, so exactly one active bit means sum(s) = r-2.
    rows = repeat(collect(eachindex(plan.offsets)); inner=plan.spec.r)
    C = sparse(rows, 1:plan.n_bits, ones(plan.n_bits), length(plan.offsets), plan.n_bits)
    return (; C=_stored(C, storage), d=Float64.(plan.resolutions .- 2))
end

function _sampling_data(base, binary, plan, storage, rounding, checkpoint=nothing)
    B, d = binary.coupling_bits, binary.linear_bits
    # Accumulate in exact dyadic arithmetic, then bound the error from
    # rounding the coefficients that actually leave Julia. A large penalty
    # must not silently erase a small original objective in the spin fields.
    w_exact = [-_exact(x)/2 for x in _checked_grid(d, checkpoint)]
    offset = _exact(binary.energy_offset) - sum(_checked_grid(w_exact, checkpoint))
    Q = copy(B)
    for (i, j, h) in _checked_grid(_upper_entries(B), checkpoint)
        quarter = _exact(h)/4
        q = _compiled_coefficient(quarter, "spin coupling ($i,$j)")
        Q[i,j] = Q[j,i] = q
        rounding += abs(_exact(q) - quarter)
        w_exact[i] -= quarter
        w_exact[j] -= quarter
        offset += quarter
    end
    spin_offset = _compiled_coefficient(offset, "spin offset")
    w = [_compiled_coefficient(x, "spin field $i") for (i, x) in _checked_grid(enumerate(w_exact), checkpoint)]
    rounding += abs(_exact(spin_offset) - offset) +
                sum(i -> abs(_exact(w[i]) - w_exact[i]), _checked_grid(eachindex(w), checkpoint))
    # Use the varying objective scale, omitting the arbitrary input constant.
    varying_scale = maximum(abs, base.linear_bits; init=0.0)
    for (_, _, h) in _checked_grid(_upper_entries(base.coupling_bits), checkpoint)
        varying_scale = max(varying_scale, abs(h))
    end
    scale = _exact(varying_scale)
    rounding <= _GRID_ATOL + _GRID_RTOL*scale && rounding <= (_GRID_ATOL + _GRID_RTOL)*scale ||
        _numerical_admission(ArgumentError("penalty or spin conversion loses objective accuracy in Float64"))
    constraints = plan.validity isa LinearValidity ? _linear_constraints(plan.spec, plan, storage) : (;)
    any(_has_nonzero, (Q, w, values(constraints)...)) ||
        _numerical_admission(ArgumentError("grid-degenerate nonconstant model has zero effective sampler coefficients"))
    return QuadraticProblem(Q, w; n_binary_vars=plan.n_bits, constraints...), spin_offset, rounding
end

# Compare encoded energies at the decoded Float64 coordinates, including
# reconstruction roundoff and cancellation between terms.
function _grid_fidelity(model, transform, normalized, exact_linear, plan, base, spin_rounding)
    if plan.spec isa OneHot
        return _onehot_qp_fidelity(model, transform, normalized, exact_linear, plan, base, spin_rounding)
    end
    r = plan.spec.r
    lower = _exact_value(model.objective, _exact.(model.lower))
    upper = lower
    anchor_value = lower
    scale_floor = abs(lower)
    error_bound = abs(lower - _exact(base.energy_offset)) + spin_rounding
    widths = _exact.(model.upper[transform.free]) - _exact.(model.lower[transform.free])
    position_errors = zero.(widths)
    for (i, coordinate) in enumerate(transform.free)
        linear = exact_linear[i]
        quadratic = _exact(model.objective.H[coordinate,coordinate]) * widths[i]^2 / 2
        anchor = _exact(model.lower[coordinate])
        lifted = term_min = term_max = local_error = zero(linear)
        for k in 1:r
            t = _exact(k)/r
            x = Float64(anchor + widths[i]*t)
            actual_t = (_exact(x) - anchor) / widths[i]
            position_errors[i] = max(position_errors[i], abs(actual_t - t))
            value = actual_t * (linear + quadratic*actual_t)
            term_min, term_max = min(term_min, value), max(term_max, value)
            scale_floor = max(scale_floor, abs(anchor_value + value))
            lifted += plan.spec isa Hamming ? _hamming_increment(base, plan.offsets[i], k) :
                      _exact(base.linear_bits[plan.offsets[i]+r-k+1])
            local_error = max(local_error, abs(lifted - value))
        end
        lower += term_min
        upper += term_max
        error_bound += local_error
    end
    for (i, j, _) in _upper_entries(normalized.H)
        i == j && continue
        a, b = transform.free[i], transform.free[j]
        pair = _exact(model.objective.H[a,b]) * widths[i] * widths[j]
        lower += min(0, pair)
        upper += max(0, pair)
        # Use a conservative scale bound from the pair's endpoint energies.
        scale_floor = max(scale_floor, abs(pair)/4)
        coupling = _exact(base.coupling_bits[plan.offsets[i]+1,plan.offsets[j]+1])
        error_bound += abs(coupling*BigInt(r)^2 - pair) +
                       abs(pair)*(position_errors[i] + position_errors[j])
    end
    min_abs = max(0, lower, -upper)
    sufficient = error_bound <= _GRID_ATOL + _GRID_RTOL*min_abs &&
                 error_bound <= _GRID_ATOL*scale_floor + _GRID_RTOL*min_abs
    return sufficient, Float64(error_bound, RoundUp)
end

function _check_small_grid(model, transform, plan, base, binary, problem, spin_offset, checkpoint=()->nothing)
    # Where interval bounds overlap zero, a small grid can still be admitted
    # exactly. Put a deterministic cap on this fallback; never silently accept
    # a large cancellation-sensitive model with an inconclusive certificate.
    r, p = plan.spec.r, length(transform.free)
    levels = _table_levels(plan.spec, r)
    states = 1
    for _ in 1:p
        states <= 1024 ÷ length(levels) ||
            _numerical_admission(ArgumentError("cannot certify whole-objective grid fidelity in Float64"))
        states *= length(levels)
    end
    encoded = QuadraticObjective(binary.coupling_bits, binary.linear_bits, binary.energy_offset)
    unpenalized = QuadraticObjective(base.coupling_bits, base.linear_bits, base.energy_offset)
    spin = QuadraticObjective(problem.Q, problem.w, spin_offset)
    evaluations = NTuple{4,Rational{BigInt}}[]
    keys = states == 1 ? (fill(only(levels), p),) : Iterators.product(ntuple(_ -> levels, p)...)
    for key in _checked_grid(keys, checkpoint)
        bits = [Int(_representative_bit(plan.spec, r, level, slot)) for level in key for slot in 1:r]
        point = reconstruct(transform, [_exact(k)/r for k in key])
        push!(evaluations, (_grid_reference(model.objective, point, checkpoint),
                            _exact_value(unpenalized, bits), _exact_value(encoded, bits),
                            _exact_value(spin, 1 .- 2 .* bits)))
    end
    scale = maximum(values -> abs(first(values)), evaluations)
    for (ordinal, values) in enumerate(evaluations), value in Base.tail(values)
        _grid_check(first(values), value, scale, "complete objective at $(_encoding_id(plan.spec)) grid point $ordinal")
    end
end

_grid_reference(obj::QuadraticObjective, x, checkpoint) = _exact_value(obj, _exact.(x))

function _artifact(snapshot, transform, normalized, plan, tables, base, binary, problem,
                   spin_offset, shortcut, storage, diagnostics)
    # Keep model identity independent of encoding and compatible with previous model digests.
    model_digest = _fingerprint(snapshot, _compiler_schema(snapshot, :unary))
    schema = _compiler_schema(snapshot, _encoding_id(plan.spec))
    payload = (snapshot, model_digest, transform, normalized, plan, tables,
               base, binary, problem, spin_offset, shortcut, storage, diagnostics)
    return CompiledQHD(payload..., _fingerprint(payload, schema))
end

"""
    compile(model; encoding, validity, storage=:sparse)

Prepare a bounded QP for coarse QIHD sampling. Supply `FoldedPenalty(weight)`
or `LinearValidity()` for `Unary(r)` and `OneHot(r)`. Use `Hamming(r)`
without a validity option. Coefficients and grid tables are owned
Float64 arrays. Exact rational intermediates protect affine substitution,
differences and spin conversion against premature cancellation. `:sparse`
uses CSC construction and CSR transport. `:dense` explicitly materializes
dense worker arrays. Compilation never launches Python.
"""
function compile(model::QHDProblem{QuadraticObjective}; encoding::GridEncoding,
                 validity=_default_validity(encoding), storage::Symbol=:sparse)
    return _native_errors(() -> _compile(model, encoding, validity, storage, () -> nothing))
end

function _compile(model::QHDProblem{QuadraticObjective}, encoding, validity, storage, checkpoint)
    storage in (:sparse, :dense) || _invalid_option(:configuration, ArgumentError("storage must be :sparse or :dense"))
    snapshot = _snapshot(model)
    transform = BoxTransform(snapshot)
    plan = _encoding_plan(encoding, validity, transform)
    normalized, shortcut, exact_linear = _normalize(snapshot, transform)
    checkpoint()
    p = length(transform.free)
    if shortcut == :none
        if encoding isa Hamming
            _check_grid_coordinates(transform, (_exact(k)/encoding.r for k in _table_levels(encoding, encoding.r)))
            tables = EncodingTable[]
            base = _hamming_binary(snapshot, normalized, transform, plan, exact_linear, storage)
        else
            tables = _qp_tables(snapshot, transform, encoding, exact_linear)
            assemble = encoding isa OneHot ? _onehot_qp_base : _binary_energies
            base = assemble(snapshot, normalized, transform, plan, tables, storage)
        end
        checkpoint()
        binary, rounding = _apply_validity(base, plan)
        problem, spin_offset, spin_rounding = _sampling_data(base, binary, plan, storage, rounding)
        checkpoint()
        certified, error_bound = _grid_fidelity(snapshot, transform, normalized, exact_linear, plan,
                                                 base, spin_rounding)
        grid_check = certified ? :interval_certificate : :enumerated_grid
        certified || _check_small_grid(snapshot, transform, plan, base, binary, problem, spin_offset)
    else
        tables = EncodingTable[]
        base = binary = problem = nothing
        spin_offset = objective_value(snapshot, _shortcut_point(transform, plan))
        grid_check, error_bound = :deterministic, 0.0
    end
    diagnostics = (policy=Symbol("qp_", _encoding_id(encoding), "_v1"), grid=_grid_convention(encoding),
                   arithmetic=:exact_rational_intermediates_float64_storage,
                   grid_atol=Float64(_GRID_ATOL), grid_rtol=Float64(_GRID_RTOL),
                   whole_objective_check=grid_check, grid_error_upper_bound=error_bound,
                   original_dimension=length(snapshot.names), free_dimension=p, bit_dimension=plan.n_bits,
                   spin_nnz=isnothing(problem) ? 0 : count(!iszero, problem.Q))
    return _artifact(snapshot, transform, normalized, plan, tables, base, binary, problem,
                     spin_offset, shortcut, storage, diagnostics)
end

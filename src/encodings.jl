abstract type GridEncoding end

function _resolution(r, name)
    !(r isa Bool) && 0 < r < typemax(Int) ||
        throw(ArgumentError("$name resolution must be a positive machine integer"))
    return Int(r)
end

"""Endpoint-inclusive grid. Valid blocks contain zeros followed by ones."""
struct Unary <: GridEncoding
    r::Int
    Unary(r::Integer) = new(_resolution(r, "unary"))
end

"""Source one-hot grid. Exactly one slot is active; slot j represents (r-j+1)/r."""
struct OneHot <: GridEncoding
    r::Int
    OneHot(r::Integer) = new(_resolution(r, "one-hot"))
end

"""Bounded-QP encoding. Every bit pattern is valid and represents its Hamming weight / r."""
struct Hamming <: GridEncoding
    r::Int
    Hamming(r::Integer) = new(_resolution(r, "Hamming"))
end

const TableEncoding = Union{Unary,OneHot}

"""Add an explicit positive weight times the encoding's block validity penalty."""
struct FoldedPenalty
    weight::Float64
    function FoldedPenalty(weight::Real)
        !(weight isa Bool) && weight > 0 || throw(ArgumentError("penalty weight must be positive"))
        new(_coefficient(weight, "penalty weight"))
    end
end

"""Send unary spin inequalities or one-hot spin equalities to the worker."""
struct LinearValidity end
const BlockValidity = Union{FoldedPenalty,LinearValidity}

_encoding_id(::Unary) = :unary
_encoding_id(::OneHot) = :onehot
_encoding_id(::Hamming) = :hamming
_grid_convention(::Unary) = :nondecreasing_endpoint_inclusive
_grid_convention(::OneHot) = :source_onehot_descending
_grid_convention(::Hamming) = :hamming_weight

# Integer grid numerators, in table order. One-hot table order is slot order.
_table_levels(::Union{Unary,Hamming}, r) = 0:r
_table_levels(::OneHot, r) = r:-1:1
_default_validity(::TableEncoding) = throw(UndefKeywordError(:validity))
_default_validity(::Hamming) = nothing

struct EncodingPlan{E<:GridEncoding,V<:Union{BlockValidity,Nothing}}
    spec::E
    validity::V
    resolutions::Vector{Int}
    offsets::Vector{Int}          # count of preceding bits
    n_bits::Int
end

function _encoding_plan(encoding, validity, transform)
    if encoding isa Hamming
        isnothing(validity) || _invalid_option(:configuration, ArgumentError("Hamming has no invalid codewords and takes no validity strategy"))
    else
        validity isa BlockValidity || _invalid_option(:configuration, ArgumentError("supply FoldedPenalty(weight) or LinearValidity()"))
    end
    p, r = length(transform.free), encoding.r
    m = try
        Base.checked_mul(p, r)
    catch err
        err isa OverflowError || rethrow()
        _invalid_option(:configuration, err)
    end
    return EncodingPlan(encoding, validity, fill(r, p), [(i-1)*r for i in 1:p], m)
end

function _shortcut_point(transform, plan)
    # The smallest source one-hot level is 1/r even for a constant objective.
    level = plan.spec isa OneHot ? 1 // plan.spec.r : 0 // 1
    _check_grid_coordinates(transform, (_exact(level),))
    return reconstruct(transform, fill(level, length(transform.free)))
end

# Evaluate validity after the spin check, and decode only after all blocks pass.
_violation(::Unary, bits) = count(j -> bits[j] && !bits[j+1], 1:length(bits)-1)
_violation(::OneHot, bits) = widen(count(bits) - 1)^2
_violation(::Hamming, bits) = 0
_decode_level(::Union{Unary,Hamming}, bits) = count(bits)
_decode_level(::OneHot, bits) = length(bits) - findfirst(bits) + 1

# Any permutation of a Hamming block of a given weight has the same energy.
_representative_bit(::Union{Unary,Hamming}, r, level, slot) = slot > r-level
_representative_bit(::OneHot, r, level, slot) = slot == r-level+1

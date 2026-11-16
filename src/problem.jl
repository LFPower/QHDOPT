const CoefficientMatrix = Union{Matrix{Float64},SparseMatrixCSC{Float64,Int}}

"""
    QuadraticProblem(Q, w; n_binary_vars=0, A=nothing, b=nothing, C=nothing, d=nothing)

Normalized QIHD problem: minimize `0.5*x'Q*x + w'x` on `[-1,1]^n`, subject
to `A*x <= b` and `C*x == d`. The first `n_binary_vars` coordinates are
binary spins. Their quadratic diagonal must be zero. Continuous coordinates
may have nonzero diagonal entries. No bounds conversion or refinement occurs.

Dense strided and sparse CSC matrices are copied; sparse inputs are never
implicitly densified. Q, A and C must use one representation per problem;
mixed representations require explicit conversion. Omitted constraints have
zero rows in the same representation as Q. Omitted right-hand
sides are zero vectors of the corresponding constraint count.
"""
struct QuadraticProblem
    Q::CoefficientMatrix
    w::Vector{Float64}
    A::CoefficientMatrix
    b::Vector{Float64}
    C::CoefficientMatrix
    d::Vector{Float64}
    n_binary_vars::Int
end

function _matrix(value, name)
    value isa AbstractMatrix{<:Real} || throw(ArgumentError("$name must be a real matrix"))
    Base.require_one_based_indexing(value)
    converted = if value isa SparseMatrixCSC
        copy(SparseMatrixCSC{Float64,Int}(value))
    elseif value isa StridedMatrix
        Matrix{Float64}(value)
    else
        throw(ArgumentError("$name must be dense strided or sparse CSC; implicit densification is disabled"))
    end
    values = converted isa SparseMatrixCSC ? nonzeros(converted) : converted
    all(isfinite, values) || throw(ArgumentError("$name must have finite Float64 coefficients"))
    return converted
end

function _vector(value, name)
    value isa AbstractVector{<:Real} || throw(ArgumentError("$name must be a real vector"))
    Base.require_one_based_indexing(value)
    converted = Float64.(collect(value))
    all(isfinite, converted) || throw(ArgumentError("$name must have finite Float64 coefficients"))
    return converted
end

_has_nonzero(x::SparseMatrixCSC) = any(!iszero, nonzeros(x))
_has_nonzero(x) = any(!iszero, x)

function QuadraticProblem(Q, w; n_binary_vars::Integer=0, A=nothing, b=nothing, C=nothing, d=nothing)
    quadratic, linear = _matrix(Q, "Q"), _vector(w, "w")
    n = length(linear)
    n > 0 && size(quadratic) == (n, n) || throw(DimensionMismatch("Q must be nonempty, square, and match w"))
    issymmetric(quadratic) || throw(ArgumentError("Q must be symmetric; coefficients are not silently symmetrized"))
    !(n_binary_vars isa Bool) && 0 <= n_binary_vars <= n || throw(ArgumentError("n_binary_vars must be an integer between zero and the problem dimension"))
    all(iszero(quadratic[i, i]) for i in 1:n_binary_vars) || throw(ArgumentError(
        "binary coordinates require a zero quadratic diagonal; remove constant spin-square terms explicitly",
    ))
    inequality = A === nothing ? (quadratic isa SparseMatrixCSC ? spzeros(0, n) : zeros(0, n)) : _matrix(A, "A")
    equality = C === nothing ? (quadratic isa SparseMatrixCSC ? spzeros(0, n) : zeros(0, n)) : _matrix(C, "C")
    for (name, matrix) in (("A", inequality), ("C", equality))
        (matrix isa SparseMatrixCSC) == (quadratic isa SparseMatrixCSC) || throw(ArgumentError(
            "$name and Q use mixed dense/CSC representations; explicitly convert the matrices to one representation",
        ))
    end
    rhs_ineq = b === nothing ? zeros(size(inequality, 1)) : _vector(b, "b")
    rhs_eq = d === nothing ? zeros(size(equality, 1)) : _vector(d, "d")
    size(inequality, 2) == n && length(rhs_ineq) == size(inequality, 1) ||
        throw(DimensionMismatch("A and b must have matching rows and A must have $n columns"))
    size(equality, 2) == n && length(rhs_eq) == size(equality, 1) ||
        throw(DimensionMismatch("C and d must have matching rows and C must have $n columns"))
    any(_has_nonzero, (quadratic, linear, inequality, rhs_ineq, equality, rhs_eq)) ||
        throw(ArgumentError("an all-zero problem has undefined upstream QIHD normalization"))
    return QuadraticProblem(quadratic, linear, inequality, rhs_ineq, equality, rhs_eq, Int(n_binary_vars))
end

"""A validated problem and QIHD settings; omitted seeds are drawn in Julia and retained in `settings`."""
struct SamplingJob
    problem::QuadraticProblem
    settings::Dict{String,Any}
end

"""Reusable requested QIHD settings. An omitted seed is drawn only when preparing a job."""
struct QIHDConfig
    n_shots::Int
    n_steps::Int
    seed::Union{Nothing,Int64}
    x64::Bool
    ballistic::Bool
    verbose::Bool
    dt::Float64
    a0::Float64
    symplectic_integration::Bool
    slow_a::Bool
    lc_pr::Float64
    constant_cons::Bool
    c0_multiplier::Float64

    function QIHDConfig(;
        n_shots::Integer=100, n_steps::Integer=10000, seed::Union{Nothing,Integer}=nothing,
        x64::Bool=false, ballistic::Bool=false, verbose::Bool=false, dt::Real=0.2,
        a0::Real=1.0, symplectic_integration::Bool=true, slow_a::Bool=true,
        lc_pr::Real=3, constant_cons::Bool=false, c0_multiplier::Real=0.5,
    )
        !(n_shots isa Bool) && 0 < n_shots <= typemax(Int) || throw(ArgumentError("n_shots must be a positive machine integer"))
        !(n_steps isa Bool) && 0 < n_steps <= typemax(Int) || throw(ArgumentError("n_steps must be a positive machine integer"))
        slow_a && n_steps <= 1000 && throw(ArgumentError("slow_a=true requires n_steps > 1000"))
        for (name, value) in (("dt", dt), ("a0", a0), ("c0_multiplier", c0_multiplier))
            !(value isa Bool) && isfinite(Float64(value)) && Float64(value) > 0 ||
                throw(ArgumentError("$name must be finite and positive in Float64"))
        end
        !(lc_pr isa Bool) && isfinite(Float64(lc_pr)) && lc_pr >= 0 && (lc_pr == 0 || Float64(lc_pr) > 0) ||
            throw(ArgumentError("lc_pr must be finite and nonnegative, with positive values representable in Float64"))
        if seed !== nothing
            !(seed isa Bool) && 0 <= seed <= typemax(Int64) || throw(ArgumentError("seed must be a nonnegative Int64"))
        end
        new(n_shots, n_steps, seed, x64, ballistic, verbose, dt, a0,
            symplectic_integration, slow_a, lc_pr, constant_cons, c0_multiplier)
    end
end

function SamplingJob(problem::QuadraticProblem, config::QIHDConfig)
    settings = Dict{String,Any}(String(k) => getfield(config, k) for k in fieldnames(QIHDConfig))
    if config.seed === nothing
        settings["seed"] = Int64(rand(RandomDevice(), UInt64) & UInt64(typemax(Int64)))
    end
    return SamplingJob(problem, settings)
end

SamplingJob(problem::QuadraticProblem; kwargs...) = SamplingJob(problem, QIHDConfig(; kwargs...))

"""One ordered sample batch and its worker, source, precision, seed, and timing metadata."""
struct SampleResult
    samples::Matrix{Float64}
    metadata::Dict{String,Any}
end

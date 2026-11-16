"""
    IpoptRefiner(; tol=1e-6, max_iter=100, feasibility_tolerance=0.0)

Refine in free original coordinates using exact first and second derivatives
and zero bound relaxation. Load the optional extension with `using QHDOPT, Ipopt`.
`feasibility_tolerance` is the absolute tolerance for validating and, when
necessary, projecting the returned point. Supplied starts must lie in the box.
"""
struct IpoptRefiner <: AbstractRefiner
    tol::Float64
    max_iter::Int
    feasibility_tolerance::Float64

    function IpoptRefiner(; tol::Real=1e-6, max_iter::Integer=100, feasibility_tolerance::Real=0.0)
        tolerance = _coefficient(tol, "Ipopt tolerance")
        !(tol isa Bool) && tolerance > 0 || throw(ArgumentError("tol must be finite and positive"))
        !(max_iter isa Bool) && 0 <= max_iter <= typemax(Cint) ||
            throw(ArgumentError("max_iter must be a nonnegative Ipopt integer"))
        feasibility = _coefficient(feasibility_tolerance, "return feasibility tolerance")
        !(feasibility_tolerance isa Bool) && feasibility >= 0 ||
            throw(ArgumentError("feasibility_tolerance must be finite and nonnegative"))
        return new(tolerance, Int(max_iter), feasibility)
    end
end

function _refiner_info(refiner::IpoptRefiner)
    ext = Base.get_extension(@__MODULE__, :QHDOPTIpoptExt)
    versions = isnothing(ext) ? nothing : ext.versions()
    options = (; tol=refiner.tol, max_iter=refiner.max_iter, hessian_approximation=:exact,
               bound_relax_factor=0.0, honor_original_bounds=false, linear_solver=:mumps,
               nlp_lower_bound_inf=-floatmax(Float64), nlp_upper_bound_inf=floatmax(Float64),
               option_file_name="", check_derivatives_for_naninf=true,
               initialization=:ipopt_default_interior, bound_push=0.01, bound_frac=0.01)
    return (; solver=:ipopt, options, versions, feasibility_tolerance=refiner.feasibility_tolerance)
end

function _run_refiner(refiner::IpoptRefiner, model::QHDProblem, start::AbstractStart)
    ext = Base.get_extension(@__MODULE__, :QHDOPTIpoptExt)
    isnothing(ext) && _invalid_option(:refinement, ArgumentError(
        "IpoptRefiner requires the optional Ipopt package. Add Ipopt to your environment and run `using QHDOPT, Ipopt`.",
    ))
    return ext.refine(refiner, model, start)
end

using QHDOPT
using Ipopt
isempty(ARGS) || error("usage: refinement.jl")
model = quadratic_problem([2.0 0.5; 0.5 1.0], [-1.5, 0.5];
    constant=1, lower=[0, -2], upper=[2, 1])
for refiner in (TNCRefiner(), IpoptRefiner(tol=1e-8))
    result = solve_classical(model; starts=[[0.0, -2.0]], refiner)
    record = only(result.refinement_records)
    println("Refiner: ", record.solver, "; status: ", record.status,
        "; usable return: ", record.usable)
    println("Selected point: ", solution(result), "; objective: ", result.selected.value)
end

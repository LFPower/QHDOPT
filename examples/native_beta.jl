using QHDOPT
length(ARGS) == 1 || error("usage: native_beta.jl <beta-python>")
python = abspath(only(ARGS))
isfile(python) || error("The selected beta Python executable does not exist.")
model = quadratic_problem([2.0 0.5; 0.5 1.0], [-1.5, 0.5];
    constant=1, lower=[0, -2], upper=[2, 1], names=[:x, :y])
prepared = compile(model; encoding=Unary(2), validity=FoldedPenalty(10), storage=:dense)
result = solve(prepared; backend=PythonQIHDBackend(; python),
    sampler=QIHDConfig(seed=7, n_shots=64, n_steps=1200))
result.raw === nothing && error("The sampler did not return a batch: $(result.status)")
metadata = result.raw.metadata
metadata["source_revision"] == "private-beta-20261115221654" ||
    error("The selected interpreter does not provide the expected beta worker.")
println("Worker: ", metadata["phisolve_version"], " on ", metadata["device"])
println("Status: ", result.status, "; valid trials: ", result.counts.valid,
    "/", result.counts.actual_trials)
if isnothing(result.selected)
    println("No usable candidate was found.")
else
    println("Candidate: ", solution(result), "; objective: ", result.selected.value)
end

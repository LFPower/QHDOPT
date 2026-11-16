using JuMP
import QHDOPT
length(ARGS) == 1 || error("usage: jump_beta.jl <beta-python>")
python = abspath(only(ARGS))
isfile(python) || error("The selected beta Python executable does not exist.")
model = Model(() -> QHDOPT.Optimizer(
    backend=QHDOPT.PythonQIHDBackend(; python),
    encoding=QHDOPT.Unary(2), validity=QHDOPT.FoldedPenalty(10),
    storage=:dense, sampler=QHDOPT.QIHDConfig(seed=7, n_shots=64, n_steps=1200)))
@variable(model, 0 <= x <= 2)
@variable(model, -2 <= y <= 1)
@objective(model, Min, 1 + x^2 + 0.5x*y + 0.5y^2 - 1.5x + 0.5y)
optimize!(model)
native = get_attribute(model, QHDOPT.NativeResult())
native === nothing && error("No solver result: $(raw_status(model))")
native.raw === nothing && error("The sampler did not return a batch.")
native.raw.metadata["source_revision"] == "private-beta-20261115221654" ||
    error("The selected interpreter does not provide the expected beta worker.")
println("Termination: ", termination_status(model), "; primal: ", primal_status(model))
if has_values(model)
    println("Candidate: ", [value(x), value(y)], "; objective: ", objective_value(model))
else
    println("No usable candidate was found.")
end

module QHDOPT

import Libdl
using LinearAlgebra: issymmetric
using NPZ: npzread, npzwrite
using Random: RandomDevice, rand
using SHA: sha256
using Serialization: serialize, deserialize
using SparseArrays: SparseMatrixCSC, nonzeros, rowvals, sparse, spzeros
using TOML

export QuadraticProblem, SamplingJob, SampleResult, WorkerError
export sample_jobs, sample_quadratic, sample_spins
export QHDProblem, quadratic_problem, structured_problem, objective_value
export Unary, OneHot, Hamming, FoldedPenalty, LinearValidity
export CompiledQHD, compile, inspect, reconstruct
export PythonQIHDBackend, QIHDConfig, NoRefiner, IpoptRefiner, QHDResult, solve, solve_classical, solution
export SolveDeadlineError
export TNCRefiner, SerialMultistart, ProcessMultistart
export success_probability, objective_summary

"""Create the optional MathOptInterface optimizer. Load JuMP or MathOptInterface first."""
function Optimizer end

"""JuMP model attribute for an owned copy of the most recent native solve result."""
function NativeResult end

include("problem.jl")
include("bridge.jl")
include("model.jl")
include("structured.jl")
include("derivatives.jl")
include("encodings.jl")
include("compiler.jl")
include("onehot.jl")
include("hamming.jl")
include("structured_tables.jl")
include("structured_compiler.jl")
include("refinement.jl")
include("ipopt.jl")
include("tnc.jl")
include("multistart.jl")
include("solve.jl")

end # module

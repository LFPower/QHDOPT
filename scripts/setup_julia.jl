using Pkg
VERSION == v"1.13.1" || error("This installation path is tested with Julia 1.13.1.")
root = normpath(joinpath(@__DIR__, ".."))
environment = joinpath(root, ".local", "julia-env")
isfile(joinpath(root, "Project.toml")) || error("Run from an intact QHDOPT checkout.")
isempty(ARGS) || error("usage: julia --startup-file=no scripts/setup_julia.jl")
Pkg.activate(environment)
Pkg.develop(path=root)
packages = [
    PackageSpec(name="NPZ", version=v"0.4.3"),
    PackageSpec(name="JuMP", version=v"1.32.1"),
    PackageSpec(name="MathOptInterface", version=v"1.54.0"),
    PackageSpec(name="Ipopt", version=v"1.16.0"),
]
Pkg.add(packages; preserve=Pkg.PRESERVE_ALL)
Pkg.pin(packages)
Pkg.instantiate()
Pkg.status()
println("QHDOPT user environment: ", environment)
println("Use --project=", environment, " for the examples and checks.")

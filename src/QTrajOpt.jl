module QTrajOpt

using RobotDynamics
using StaticArrays
using ForwardDiff, FiniteDiff
using LinearAlgebra
const RD = RobotDynamics

const SA_C64 = SA{ComplexF64}

export SA_C64

include("utils.jl")
include("spin.jl")
include("bilinear_dynamics.jl")
include("dynamics.jl")
include("gates.jl")

end # module


using QTrajOpt
using StaticArrays
using RobotDynamics
using BenchmarkTools
using ForwardDiff
using Test
const RD = RobotDynamics
const QTO = QTrajOpt

n, m = 32,1
A = randn(n, n)
B = [randn(n, n) for i = 1:m]
C = randn(n, m)
model = QTO.BiLinearDynamics{n,m}(A, B, C)
xs,us = rand(model)
x,u = Vector(xs), Vector(us)

xdot = zeros(n)
RD.dynamics!(model, xdot, x, u)
@test xdot ≈ RD.dynamics(model, xs, us)

f(z) = RD.dynamics(model, z[1:n], z[n+1:n+m])
z = [x;u]
J = zeros(n, n+m)
RD.jacobian!(model, J, xdot, x, u)
@test J ≈ ForwardDiff.jacobian(f, z)

# ControlIntegralDerivative model
umodel = QTO.ControlIntegralDerivative(model)
RD.state_dim(umodel) == n + 3m
RD.control_dim(umodel) == m
n2,m2 = RD.dims(umodel)
∫u = randn(m)
du = randn(m)
ddu = randn(m)

x2 = [x; ∫u; u; du]
u2 = ddu

xdot2 = zeros(n2)
RD.dynamics!(umodel, xdot2, x2, u2)
xdot2 ≈ [xdot; u; du; ddu]

f2(y,z) = RD.dynamics!(umodel, y, z[1:n2], z[n2+1:n2+m2])
z2 = [x2; u2]
J0 = zeros(n2, n2 + m2)
ForwardDiff.jacobian!(J0, f2, xdot2, z2)

J2 = zero(J0)
RD.jacobian!(umodel, J2, xdot, x2, u2)
@test J2 ≈ J0


# Try a discrete version
dmodel = RD.DiscretizedDynamics{RD.RK3}(umodel)
dt = 0.1
z2 = KnotPoint{n2,m2}(x2, u2, 0.0, dt)
RD.discrete_dynamics!(dmodel, xdot2, z2)
RD.jacobian!(RD.InPlace(), RD.UserDefined(), dmodel, J2, xdot2, z2)


# Original dynamics
model0 = QTrajOpt.TwoQubit(2pi, 2pi)  # convert frequencies to rad/ns
J0 .= 0
RD.jacobian!(RD.InPlace(), RD.ForwardAD(), model0, J0, xdot2, z2)
J2 .= 0
RD.jacobian!(RD.InPlace(), RD.UserDefined(), model0, J2, xdot2, z2)
J0 ≈ J2

@btime RD.jacobian!(RD.InPlace(), RD.ForwardAD(), $model0, $J2, $xdot2, $z2)
@btime RD.jacobian!(RD.InPlace(), RD.UserDefined(), $model0, $J2, $xdot2, $z2)

# Use Exponential Integrator
dmodel0 = QTrajOpt.DiscreteTwoQubit(model0)
RD.discrete_dynamics!(dmodel0, xdot2, z2)
RD.jacobian!(RD.InPlace(), RD.ForwardAD(), dmodel0, J0, xdot2, z2)

# Use RK3
rk3 = RD.DiscretizedDynamics{RD.RK3}(model0)
RD.discrete_dynamics!(rk3, xdot2, z2)
RD.jacobian!(RD.InPlace(), RD.UserDefined(), rk3, J2, xdot2, z2)

@btime RD.jacobian!(RD.InPlace(), RD.ForwardAD(), $dmodel0, $J0, $xdot2, $z2)
@btime RD.jacobian!(RD.InPlace(), RD.UserDefined(), $rk3, $J2, $xdot2, $z2)
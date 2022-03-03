using Altro
using TrajectoryOptimization
using RobotDynamics
using ForwardDiff
using FiniteDiff
const TO = TrajectoryOptimization
const RD = RobotDynamics

WDIR = joinpath(@__DIR__, "..", "..", "rbqoc")
include(joinpath(WDIR, "src", "twospin", "twospin_minimal_base.jl"))

# problem
const CONTROL_COUNT = 1
const STATE_COUNT = 4
const ASTATE_SIZE = STATE_COUNT * HDIM_TWOSPIN_ISO + 3 * CONTROL_COUNT
const ACONTROL_SIZE = CONTROL_COUNT
# state indices
const STATE1_IDX = 1:HDIM_TWOSPIN_ISO
const STATE2_IDX = STATE1_IDX[end] + 1:STATE1_IDX[end] + HDIM_TWOSPIN_ISO
const STATE3_IDX = STATE2_IDX[end] + 1:STATE2_IDX[end] + HDIM_TWOSPIN_ISO
const STATE4_IDX = STATE3_IDX[end] + 1:STATE3_IDX[end] + HDIM_TWOSPIN_ISO

const INTCONTROLS_IDX = STATE4_IDX[end] + 1:STATE4_IDX[end] + CONTROL_COUNT
const CONTROLS_IDX = INTCONTROLS_IDX[end] + 1:INTCONTROLS_IDX[end] + CONTROL_COUNT
const DCONTROLS_IDX = CONTROLS_IDX[end] + 1:CONTROLS_IDX[end] + CONTROL_COUNT
# control indices
const D2CONTROLS_IDX = 1:CONTROL_COUNT

# model
RD.@autodiff struct QubitModel <: RD.DiscreteDynamics end
@inline RD.state_dim(::QubitModel) = ASTATE_SIZE
@inline RD.control_dim(::QubitModel) = ACONTROL_SIZE

# dynamics
function RD.discrete_dynamics(model::QubitModel, astate,
                              acontrol, time, dt)
    h_prop = exp(dt * (NEGI_H0_TWOSPIN_ISO
     + astate[CONTROLS_IDX[1]] * NEGI_H1_TWOSPIN_ISO_3))
    state1 = h_prop * astate[STATE1_IDX]
    state2 = h_prop * astate[STATE2_IDX]
    state3 = h_prop * astate[STATE3_IDX]
    state4 = h_prop * astate[STATE4_IDX]
    intcontrols = astate[INTCONTROLS_IDX] + astate[CONTROLS_IDX] * dt
    controls = astate[CONTROLS_IDX] + astate[DCONTROLS_IDX] * dt
    dcontrols = astate[DCONTROLS_IDX] + acontrol[D2CONTROLS_IDX] * dt
    astate_ = [
        state1; state2; state3; state4; intcontrols; controls; dcontrols;
    ]

    return astate_
end

function build_prob(;gate_type=sqrtiswap, evolution_time=70., solver_type=altro,
                  sqrtbp=false, integrator_type=rk3, smoke_test=false,
                  dt_inv=Int64(1e1), constraint_tol=1e-8, al_tol=1e-4,
                  pn_steps=2, max_penalty=1e11, verbose=true, save=true,
                  max_iterations=Int64(2e5),
                  max_cost_value=1e8, qs=[1e0, 1e0, 1e0, 1e-1, 1e-1],
                  benchmark=false, projected_newton=true)
    # model configuration
    model = QubitModel()
    n = state_dim(model)
    m = control_dim(model)
    t0 = 0.

    # initial state
    x0 = zeros(n)
    x0[STATE1_IDX] = TWOSPIN_ISO_1
    x0[STATE2_IDX] = TWOSPIN_ISO_2
    x0[STATE3_IDX] = TWOSPIN_ISO_3
    x0[STATE4_IDX] = TWOSPIN_ISO_4
    x0 = SVector{n}(x0)

    # final state
    (target_state1, target_state2,
     target_state3, target_state4) = target_states(gate_type)
    xf = zeros(n)
    xf[STATE1_IDX] = target_state1
    xf[STATE2_IDX] = target_state2
    xf[STATE3_IDX] = target_state3
    xf[STATE4_IDX] = target_state4
    xf = SVector{n}(xf)

    # control amplitude constraint
    x_max = fill(Inf, n)
    x_max[CONTROLS_IDX] .= 0.5
    x_max = SVector{n}(x_max)
    x_min = fill(-Inf, n)
    x_min[CONTROLS_IDX] .= -0.5
    x_min = SVector{n}(x_min)

    # control amplitude constraint at boundary
    x_max_boundary = fill(Inf, n)
    x_max_boundary[CONTROLS_IDX] .= 0
    x_max_boundary = SVector{n}(x_max_boundary)
    x_min_boundary = fill(-Inf, n)
    x_min_boundary[CONTROLS_IDX] .= 0
    x_min_boundary = SVector{n}(x_min_boundary)

    # initial trajectory
    dt = dt_inv^(-1)
    N = Int(floor(evolution_time * dt_inv)) + 1
    U0 = [SVector{m}(
        fill(1e-4, CONTROL_COUNT)
    ) for k = 1:N-1]
    X0 = [SVector{n}([
        fill(NaN, n);
    ]) for k = 1:N]
    Z = SampledTrajectory(X0, U0, dt=dt)

    # cost function
    Q = Diagonal(SVector{n}([
        fill(qs[1], STATE_COUNT * HDIM_TWOSPIN_ISO); # ψ1, ψ2, ψ3, ψ4
        fill(qs[2], CONTROL_COUNT); # ∫a
        fill(qs[3], CONTROL_COUNT); # a
        fill(qs[4], CONTROL_COUNT); # ∂a
    ]))
    Qf = Q * N
    R = Diagonal(SVector{m}([
        fill(qs[5], CONTROL_COUNT); # ∂2a
    ]))
    objective = LQRObjective(Q, R, Qf, xf, N)

    # constraints
    # must satisfy control amplitude bound
    control_bnd = BoundConstraint(n, m, x_max=x_max, x_min=x_min)
    # must statisfy conrols start and end at 0
    control_bnd_boundary = BoundConstraint(n, m, x_max=x_max_boundary, x_min=x_min_boundary)
    # must reach target state, must have zero net flux
    target_astate_constraint = GoalConstraint(xf, [STATE1_IDX; STATE2_IDX;
                                                   STATE3_IDX; STATE4_IDX;
                                                   INTCONTROLS_IDX])
    # must obey unit norm
    norm_constraints = [NormConstraint(n, m, 1, TO.Equality(), idx)
                        for idx in [STATE1_IDX, STATE2_IDX, STATE3_IDX, STATE4_IDX]]

    constraints = ConstraintList(n, m, N)
    add_constraint!(constraints, control_bnd, 2:N-2)
    add_constraint!(constraints, control_bnd_boundary, N-1:N-1)
    add_constraint!(constraints, target_astate_constraint, N:N);
    for norm_constraint in norm_constraints
        add_constraint!(constraints, norm_constraint, 2:N-1)
    end

    # solve problem
    prob = Problem(model, objective, constraints,
                                            x0, xf, Z, N, t0, evolution_time)
    return prob
end

prob0 = build_prob()


import Pkg; Pkg.activate(@__DIR__); Pkg.instantiate();
using QTrajOpt
const QTO = QTrajOpt

function TwoQubitProblem()
    # Params
    tf = 70.0  # total time (nsec) (typically 20-50)
    dt = 0.1   # time step (nsec)
    # ω0 = 0.04  # GHz 
    # ω1 = 0.06  # GHz 
    ω0 = 1.0
    ω1 = 1.0

    # Model
    model = QTrajOpt.TwoQubit(2pi*ω0, 2pi*ω1)  # convert frequencies to rad/ns
    dmodel = QTrajOpt.DiscreteTwoQubit(model)
    n,m = RD.dims(model)
    N = round(Int, tf / dt) + 1

    # Initial and Final states
    q0 = [
        SA_C64[1,0,0,0],  # i.e. 0b00
        SA_C64[0,1,0,0],
        SA_C64[0,0,1,0],
        SA_C64[0,0,0,1],
    ]
    x0 = zeros(n)
    for i = 1:4
        QTO.setqstate!(model, x0, q0[i], i)
    end

    U = QTrajOpt.sqrtiSWAP()
    xf = zeros(n)
    for i = 1:4
        QTO.setqstate!(model, xf, U*q0[i], i)
    end

    # Cost Function
    qs = [1e0, 1e0, 1e0, 1e-1, 1e-1]
    nqubits = QTO.nqubits(model)     # number of qubits
    qubitsize = QTO.qubitdim(model)  # size of a single qubit
    qudim = nqubits * qubitsize 
    ncontrol = RD.control_dim(model)
    
    Q = Diagonal(SVector{n}([
        fill(qs[1], qudim); # ψ1, ψ2, ψ3, ψ4
        fill(qs[2], ncontrol); # ∫a
        fill(qs[3], ncontrol); # a
        fill(qs[4], ncontrol); # ∂a
    ]))
    Qf = Q * N
    R = Diagonal(SVector{m}([
        fill(qs[5], ncontrol); # ∂2a
    ]))
    obj = LQRObjective(Q, R, Qf, xf, N)

    cons = ConstraintList(n, m, N)

    # Control amplitude constraint 
    control_idx = qudim + 2
    x_max = fill(Inf, n)
    x_min = fill(-Inf, n)
    x_max[control_idx] = 0.5
    x_min[control_idx] = -0.5
    control_bound = BoundConstraint(n, m, x_max=x_max, x_min=x_min)
    add_constraint!(cons, control_bound, 1:N-1)
    
    # Initial guess
    u0 = @SVector fill(1e-4, 1)
    U0 = [copy(u0) for k = 1:N-1]
    X0 = [copy(x0) for k = 1:N]

    # Build the problem
    prob = Problem(dmodel, obj, SVector{n}(x0), tf, xf=xf, X0=X0, U0=U0)

    return prob
end

prob = TwoQubitProblem()
prob.x0 ≈ prob0.x0
prob.xf ≈ prob0.xf
prob.N == prob0.N
RD.controls(prob0) ≈ RD.controls(prob)
RD.state(prob.Z[1])
rollout!(RD.InPlace(), prob)
rollout!(prob0)
RD.state(prob.Z[end]) ≈ RD.state(prob0.Z[end])
cost(prob) ≈ cost(prob0)

opts = SolverOptions(verbose = 4, projected_newton = false, constraint_tolerance = 1e-2, cost_tolerance = 1e-1)
solver = ALTROSolver(prob, opts)
Altro.solve!(solver)
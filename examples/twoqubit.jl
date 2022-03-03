import Pkg; Pkg.activate(@__DIR__); Pkg.instantiate();
using Altro
using QTrajOpt
using TrajectoryOptimization
using RobotDynamics
using LinearAlgebra
using StaticArrays
using Random
const RD = RobotDynamics
const QTO = QTrajOpt 
const TO = TrajectoryOptimization

function TwoQubitProblem(;
    tf=70.0,  # evolution time (nsec)
    dt=0.1,   # time step (nsec)
)
    # Params
    # ω0 = 0.04  # GHz 
    # ω1 = 0.06  # GHz 
    ω0 = 1.0
    ω1 = 1.0

    # Model
    model = QTrajOpt.TwoQubit(2pi*ω0, 2pi*ω1)  # convert frequencies to rad/ns
    dmodel = RD.DiscretizedDynamics{RD.RK3}(model) 
    # dmodel = QTrajOpt.DiscreteTwoQubit(model)
    n,m = RD.dims(model)
    nquantum = n - 3  # number of state elements for the quantum information 
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
    
    Qd = [
        fill(qs[1], qudim); # ψ1, ψ2, ψ3, ψ4
        fill(qs[2], ncontrol); # ∫a
        fill(qs[3], ncontrol); # a
        fill(qs[4], ncontrol); # ∂a
    ]
    Q = Diagonal(SVector{n}(Qd))
    Qdf = Qd * N 
    # Qdf[nquantum + 1] *= 10
    Qf = Diagonal(SVector{n}(Qdf))
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
    # add_constraint!(cons, control_bound, 1:N-1)

    # Goal constraint
    goal = GoalConstraint(xf, 1:nquantum)
    # goal = GoalConstraint(xf, nquantum .+ (1:1))
    add_constraint!(cons, goal, N)

    # Norm Constraint
    norm_cons = map(1:nqubits) do i
        NormConstraint(n, m, 1.0, TO.Equality(), QTO.getqstateinds(model, i))
    end
    for con in norm_cons
        # add_constraint!(cons, con, 1:N)
    end
    
    # Initial guess
    u0 = @SVector fill(1e-4, 1)
    U0 = [copy(u0) for k = 1:N-1]
    X0 = [copy(x0) for k = 1:N]

    # Build the problem
    prob = Problem(dmodel, obj, SVector{n}(x0), tf, xf=xf, X0=X0, U0=U0, constraints=cons)

    return prob
end

prob = TwoQubitProblem(tf=20.)
solver = ALTROSolver(prob)
solver.opts.dynamics_funsig = RD.InPlace()
solver.opts.penalty_initial = 1e-3
solver.opts.cost_tolerance_intermediate = 1e-1
solver.opts.iterations = 2000
solver.opts.cost_tolerance = 1e-3
solver.opts.verbose = 4
# solver.opts.dynamics_diffmethod = RD.UserDefined()
solve!(solver)
prob.xf[1:32]
xf = RD.states(solver)[end]
norm(xf[1:32] - prob.xf[1:32])

using Plots
RD.dims(solver)
plot(RD.states(get_trajectory(solver), 34))
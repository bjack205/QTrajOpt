using Altro
using QTrajOpt
using TrajectoryOptimization
using RobotDynamics
using LinearAlgebra
using StaticArrays
const RD = RobotDynamics

function TwoQubitProblem()
    # Params
    tf = 20.0  # total time (nsec) (typically 20-50)
    dt = 0.2   # time step (nsec)
    # ω0 = 0.04  # GHz 
    # ω1 = 0.06  # GHz 
    ω0 = 1.0
    ω1 = 1.0

    # Model
    model = QTrajOpt.TwoQubit(2pi*ω0, 2pi*ω1)  # convert frequencies to rad/ns
    n,m = size(model)
    N = round(Int, tf / dt) + 1

    # Initial and Final states
    x0_1 = SA_C64[1,0,0,0]  # i.e. 0b00
    x0_2 = SA_C64[0,1,0,0]
    x0_3 = SA_C64[0,0,1,0]
    x0_4 = SA_C64[0,0,0,1]
    x0 = [x0_1; x0_2; x0_3; x0_4]
    x0 = [real(x0); imag(x0)]

    U = QTrajOpt.sqrtiSWAP()
    xf_1 = U*x0_1
    xf_2 = U*x0_2
    xf_3 = U*x0_3
    xf_4 = U*x0_4
    xf = [xf_1; xf_2; xf_3; xf_4]
    xf = [real(xf); imag(xf)]

    # Cost Function
    Q = Diagonal(@SVector fill(1.0, n))
    R = Diagonal(@SVector fill(0.01, m)) 
    Qf = 100 * Q
    obj = LQRObjective(Q, R, Qf, xf, N)

    # Build the problem
    prob = Problem(model, obj, x0, tf, xf=xf, integration=RD.RK4(model)) 

    # Initial guess
    u0 = @SVector fill(0.0, 1)
    initial_controls!(prob, u0)

    return prob
end
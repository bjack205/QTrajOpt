

RD.@autodiff struct TwoQubit <: RobotDynamics.ContinuousDynamics
    ω0::Float64
    ω1::Float64
end
TwoQubit() = TwoQubit(1.0, 1.0)
RobotDynamics.control_dim(::TwoQubit) = 1
RobotDynamics.state_dim(::TwoQubit) = 8 * 4 + 3
nqubits(::TwoQubit) = 4
qubitdim(::TwoQubit) = 8

function getdrifthamiltonian(model::TwoQubit)
    fq_1 = 0.047
    fq_2 = 0.060
    ω1 = fq_1 * 2pi
    ω2 = fq_2 * 2pi

    I2 = SA[1 0; 0 1]
    σz = paulimat(:z)
    σz_1 = kron(σz, I2)
    σz_2 = kron(I2, σz)
    H = σz_1 * ω1 / 2 + σz_2 * ω2 / 2
    return H
end

function getdrivehamiltonian(::TwoQubit)
    I2 = SA[1 0; 0 1]
    σx = paulimat(:x)
    σx_1 = kron(σx, I2)
    σx_2 = kron(I2, σx)
    Hdrive = σx_1 * σx_2 * 2pi 
    return Hdrive
end

function getqstateinds(model::TwoQubit, i::Int)
    nqstates = 4
    ir = (i-1)*2nqstates
    return ir .+ (1:8)
end

function getqstate(model::TwoQubit, x, i::Int)
    nqstates = 4
    # ir = (i-1)*nqstates
    # ic = ir + nqstates^2
    ir = (i-1)*2nqstates
    ic = ir + nqstates
    SA[
        x[ir+1] + x[ic+1]*1im,
        x[ir+2] + x[ic+2]*1im,
        x[ir+3] + x[ic+3]*1im,
        x[ir+4] + x[ic+4]*1im,
    ] 
end

function setqstate!(model::TwoQubit, x, ψ, i::Int)
    @assert length(ψ) == 4
    nqstates = 4
    # ir = (1:4) .+ (i-1)*nqstates
    # ic = ir .+ nqstates^2
    ir = (1:4) .+ (i-1)*2nqstates
    ic = ir .+ nqstates
    x[ir] .= real(ψ)
    x[ic] .= imag(ψ)
    return x
end

function RobotDynamics.dynamics!(model::TwoQubit, xdot, x, u)
    ω0, ω1 = model.ω0, model.ω1
    ħ = 1.0  # Plancks Constant

    # Extract controls
    nqstates = 4
    nstates_quantum = nqstates^2 * 2  # number of state elements describing quantum bits
    ∫a = x[nstates_quantum + 1]
    a = x[nstates_quantum + 2]
    da = x[nstates_quantum + 3]
    dda = u[1]

    # Form the Hamiltonian
    H_drift = Diagonal(SA[ω0 + ω1, ω0 - ω1, -ω0 + ω1, -ω0 - ω1]) / 2
    H_drive = SA[
        0 0 0 1
        0 0 1 0
        0 1 0 0
        1 0 0 0
    ] / 2 * a 
    # H_drift = getdrifthamiltonian(model)
    # H_drive = getdrivehamiltonian(model)
    H = (H_drift + H_drive) * (-im)


    # Compute the dynamics
    nqstates = 4  # number of quantum states
    for i = 1:nqstates
        ψi = getqstate(model, x, i)
        ψdot = H*ψi
        setqstate!(model, xdot, ψdot, i)
    end
    xdot[nstates_quantum + 1] = a
    xdot[nstates_quantum + 2] = da
    xdot[nstates_quantum + 3] = dda
    return nothing
end

function RobotDynamics.jacobian!(model::TwoQubit, J, xdot, x, u)
    J .= 0
    ω0, ω1 = model.ω0, model.ω1

    # Extract controls
    nqstates = 4
    nstates_quantum = nqstates^2 * 2  # number of state elements describing quantum bits
    ∫a = x[nstates_quantum + 1]
    a = x[nstates_quantum + 2]
    da = x[nstates_quantum + 3]
    dda = u[1]

    # Form the Hamiltonian
    H_drift = Diagonal(SA[ω0 + ω1, ω0 - ω1, -ω0 + ω1, -ω0 - ω1]) / 2
    H_drive = SA[
        0 0 0 1
        0 0 1 0
        0 1 0 0
        1 0 0 0
    ] / 2 
    # H_drift = getdrifthamiltonian(model)
    # H_drive = getdrivehamiltonian(model)
    H = -im * (H_drift + H_drive * a)
    Hiso = complex2real(H)
    for i = 1:nqstates
        ir = (i-1)*2nqstates
        iψ =  ir .+ (1:8)
        ψi = getqstate(model, x, i)
        J[iψ, iψ] .= Hiso
        J[iψ, nstates_quantum + 2] = complex2real(-im * H_drive * ψi)
    end
    for i = 1:3
        J[nstates_quantum + i, nstates_quantum + 1 + i] = 1
    end
    return nothing
end

RD.@autodiff struct DiscreteTwoQubit <: RobotDynamics.DiscreteDynamics
    continuous_model::TwoQubit
end
DiscreteTwoQubit(args...) = DiscreteTwoQubit(TwoQubit(args...))
@inline RD.state_dim(model::DiscreteTwoQubit) = RD.state_dim(model.continuous_model)
@inline RD.control_dim(model::DiscreteTwoQubit) = RD.control_dim(model.continuous_model)

function RD.discrete_dynamics!(model::DiscreteTwoQubit, xn, x, u, t, dt)
    cmodel = model.continuous_model

    # Extract controls
    nqstates = 4
    nstates_quantum = nqstates^2 * 2  # number of state elements describing quantum bits
    ∫a = x[nstates_quantum + 1]
    a = x[nstates_quantum + 2]
    da = x[nstates_quantum + 3]
    dda = u[1]

    Hdrift = getdrifthamiltonian(cmodel)
    Hdrive = getdrivehamiltonian(cmodel)
    H = (Hdrift + Hdrive * a) * (-im)
    Hiso = complex2real(H)
    Hprop = real2complex(exp(Hiso * dt))
    # display(complex2real(Hprop))

    # Calculate the dynamics
    # ψ1 = getqstate(cmodel, x, 1)
    # x1 = complex2real(ψ1)
    # display(complex2real(Hprop) * x1)
    for i = 1:nqstates
        ψi = getqstate(cmodel, x, i)
        ψn = Hprop * ψi 
        # @show ψn
        setqstate!(cmodel, xn, ψn, i)
    end
    # Use Euler for controls
    xn[nstates_quantum + 1] = ∫a + a * dt
    xn[nstates_quantum + 2] = a  + da * dt
    xn[nstates_quantum + 3] = da + dda * dt
    return nothing
end
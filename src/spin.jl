"""
    paulimat(ax::Symbol)

Generate one of the Pauli Spin matrices. Axis `ax` must be one of `:x`, `:y`, or `:z`.
"""
function paulimat(ax::Symbol)
    if ax == :x
        return SA_C64[0 1; 1 0]
    elseif ax == :y
        return SA_C64[0 -1im; 1im 0]
    elseif ax == :z
        return SA_C64[1 0; 0 -1]
    else
        error("$ax not a recognized axis for Pauli matrices. Should be one of [:x,:y,:z].")
    end
end

"""
    paulimat(ax::Symbol, i, p)

Generate a spin matrix for the `i`th qubit with `p` qubits.
"""
paulimat(ax::Symbol, i::Integer, p::Integer) = tensormat(paulimat(ax), i, p)


"""
    tensormat(A, i, p)

For the tensor product `I₂ ⊗ … ⊗ I₂ ⊗ A ⊗ I₂ ⊗ … ⊗ I₂`
with `p` terms and `A` as term `i`. 

For example, `tensormat(A,1,2)` calculates `A ⊗ I₂` and `tensormag(A,2,3)` calculates
`I₂ ⊗ A ⊗ I₂`.
"""
function tensormat(A, i::Integer, p::Integer)
    σ0 = SMatrix{2,2,ComplexF64}(A) 
    I2 = SA_C64[1 0; 0 1]
    σ = i == 1 ? σ0 : I2
    for j = 2:p
        σi = j == i ? σ0 : I2
        σ = kron(σ, σi)
    end
    return σ
end
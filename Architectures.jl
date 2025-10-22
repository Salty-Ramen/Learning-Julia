# This is meant to contain all the test architectures for training a PINN on an
# ODE

using Lux, NNlib


struct SIR_params{T<:AbstractFloat}
    β::T
    γ::T
end

# y: 3×B (rows = S,I,R; columns = batch/time 
#=
function SIR_model(y::AbstractMatrix{<:Real}, p::SIR_params)
    @assert size(y, 1) == 3 "SIR_model expects y to be 3×B (rows=S,I,R)."
    @views begin
        S = y[1, :]; I = y[2, :]
        dS = -p.β .* S .* I
        dI =  p.β .* S .* I .- p.γ .* I
        dR =  p.γ .* I
    end
    return permutedims(hcat(dS, dI, dR))  # 3×B
end
=#

# Grey-box: supply g and an architecture that maps (y,g,p) -> 3×B RHS
function SIR_model(y::AbstractMatrix{<:Real}, p::SIR_params,
                   g::Union{AbstractVecOrMat{<:Real}, Nothing},
                   architecture::Function)
    isnothing(g) && error("Provide g or call the 2-arg SIR_model(y,p).")

    # Needs a 3xB input
    # We use the @assert macro as a sanity check.
    # This just ensures the expected dimensions while proceeding
    @assert size(y, 1) == 3

    # Force g to 1×B for consistent algebra
    # g_row = size(g, 1) == 1 ? g : reshape(vec(g), 1, :)

    @assert size(g, 2) == size(y, 2) "g must have one value per column of y."

    return architecture(y, g, p)       # must return 3×B
end

#--------Neural Nets--------
# State predictor
make_state_MLP(; hidden = (10,10), act = tanh, final_act = NNlib.softplus) = Lux.Chain(
    Lux.Dense(1, hidden[1], act),
    Lux.Dense(hidden[1], hidden[2], act),
    Lux.Dense(hidden[2], 3, final_act)
)

# Unknown learner default 
make_g_MLP(; hidden = (5,), act = tanh, final_act = tanh) = Lux.Chain(
    Lux.Dense(1, hidden[1], act),
    Lux.Dense(hidden[1], 1, final_act)
)

using Logging

# default architecture
function default_architecture(y::AbstractMatrix{<:Real}, g::AbstractMatrix{<:Real}, p::SIR_params)
    # Sanity Checks
    @assert size(g, 1) == 1 "g must be 1xB"
    @warn size(g,2) > 1 "g has more states than needed, using only the first"

    @views begin
        I = y[2,:]
        
        dS = - g[1,:]
        dI = g[1,:] - p.γ .* I
        dR = p.γ .* I
    end
    
    return permutedims(hcat(dS, dI, dR))
end

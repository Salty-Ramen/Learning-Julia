using Random, Statistics
using Lux, NNlib
using ComponentArrays
using Optimization, OptimizationOptimisers
using Zygote
using Plots
using CSV, DataFrames

#--------Load Dataset--------

function load_dataset_from_csv(path::AbstractString; df_colnames = ["t", "S", "I", "R"], y0_init = Float32.([0.999, 0.001, 0.0]))

    df = CSV.read(path, DataFrame)
    # Ref here prevents weird broadcasting issues
    @assert all(df_colnames .∈ Ref(names(df))) "CSV must have all the columns"
    t_train = Float32.(df.t)
    Y_train = permutedims(Matrix(df[:, [:S, :I, :R]])) |> x -> Float32.(x)
    y0_obs = y0_init

    Y_train_std = Statistics.std(Y_train, dims = 2)

    return t_train, Y_train, y0_obs, Y_train_std

end

#--------Configure--------

include("Data_gen.jl")
include("Architectures.jl")
rng = Random.MersenneTwister(42)

csv_path = joinpath(@__DIR__, "Results", "NoisyData.csv")

t_train, Y_train, y0_obs, Y_train_std = load_dataset_from_csv(csv_path)

t_dense = Float32.(range(first(t_train), last(t_train), length=200) |> collect)

State_MLP = make_state_MLP()
ps_StateMLP, st_StateMLP = Lux.setup(rng, State_MLP)

g_MLP = make_g_MLP()
ps_gMLP, st_gMLP = Lux.setup(rng, g_MLP)


guess_SIR_params = SIR_params(0.1f0, 0.1f0)

λ_ic  = 1.0f0 * 1000
λ_ode = 1.0f0 * 2
λ_Data = 5.5f0

trainable_params = ComponentArrays.ComponentArray(
    StateMLP = ps_StateMLP,
    gMLP     = ps_gMLP,
    ODE_par  = (β = guess_SIR_params.β, γ = guess_SIR_params.γ),
    hyperparams = (λ_ode = λ_ode, λ_Data = λ_Data)
)

MSE(ŷ, y) = Statistics.mean(abs2, vec(ŷ .- y))

function MSE(ŷ, y, denom)  # normalized MSE
    t = vec((ŷ .- y) ./ denom)
    Statistics.mean(abs2, t)
end

#--------Training--------

ctx_stage1 = (
    State_MLP   = State_MLP,
    st_StateMLP = st_StateMLP,
    t_train     = t_train,
    Y_train     = Y_train,
    Y_train_std = Y_train_std,
    y0_obs      = y0_obs,
    λ_ic        = λ_ic,
)

ctx_stage2 = (
    State_MLP   = State_MLP,
    st_StateMLP = st_StateMLP,
    g_MLP       = g_MLP,
    st_gMLP     = st_gMLP,
    t_train     = t_train,
    Y_train     = Y_train,
    Y_train_std = Y_train_std,
    y0_obs      = y0_obs,
    λ_ic        = λ_ic,
    t_dense     = t_dense,
)

# Supervised loss (fit to data)
function loss_supervised(ps, ctx)
    smodel = Lux.StatefulLuxLayer(ctx.State_MLP, ps.StateMLP, ctx.st_StateMLP)
    ŷ = smodel(ctx.t_train')
    data_mse = MSE(ŷ, ctx.Y_train, ctx.Y_train_std)
    ic_mse   = MSE(ŷ[:, 1], ctx.y0_obs, ctx.Y_train_std)
    return data_mse + ctx.λ_ic * ic_mse
end

using ForwardDiff

# 3×B Jacobian of smodel wrt scalar time, computed columnwise
dNNdt_fd(smodel, tvec::AbstractVector) = begin
    cols = map(tn -> begin
        J = ForwardDiff.jacobian(x -> smodel(reshape(x, 1, :)), [tn])  # 3×1
        vec(J)  # 3
    end, tvec)
    reduce(hcat, cols)  # 3×B
end


# ODE-regularized loss (data + residual + IC)
# SIR_model from architecture.jl supports a custom architecture via `g` and `architecture`
function loss_unsupervised(ps, ctx)
    smodel = Lux.StatefulLuxLayer(ctx.State_MLP, ps.StateMLP, ctx.st_StateMLP)
    gmodel = Lux.StatefulLuxLayer(ctx.g_MLP,     ps.gMLP,     ctx.st_gMLP)

    SIR_par = SIR_params(ps.ODE_par.β, ps.ODE_par.γ)

    Tdense = ctx.t_dense  # keep as Vector{Float32}

    dNNdt = dNNdt_fd(smodel, Tdense)              # 3×B
    f_ŷ   = SIR_model(smodel(Tdense'), SIR_par;   # 3×B
                      g = gmodel(Tdense'), architecture = default_architecture)

    ode_mse  = MSE(dNNdt, f_ŷ, ctx.Y_train_std)
    data_mse = MSE(smodel(ctx.t_train'), ctx.Y_train, ctx.Y_train_std)
    ic_mse   = MSE(smodel([ctx.t_train[1]]), ctx.y0_obs, ctx.Y_train_std)

    return ps.hyperparams.λ_ode * ode_mse + ps.hyperparams.λ_Data * data_mse + ctx.λ_ic * ic_mse
end

optfun1 = Optimization.OptimizationFunction(
    (θ, p) -> loss_supervised(θ, p),
    Optimization.AutoZygote()
)

prob1 = Optimization.OptimizationProblem(
    optfun1, 
    trainable_params, 
    ctx_stage1
)
res1 = Optimization.solve(
    prob1,
    OptimizationOptimisers.Adam(1e-03);
    maxiters = 3_000
)

trainable_params_post_stage1 = res1.u

optfun2 = Optimization.OptimizationFunction(
    (θ, p) -> loss_unsupervised(θ, p),
    Optimization.AutoZygote()
)

prob2 = Optimization.OptimizationProblem(
    optfun2,
    trainable_params_post_stage1, 
    ctx_stage2
)

res2 = Optimization.solve(
    prob2, 
    OptimizationOptimisers.Adam(5e-04); 
    maxiters=5_000
)
ps_trained = res2.u


#--------Plots--------



# Build a stateful model bound to the trained params
smodel = Lux.StatefulLuxLayer(State_MLP, ps_trained.StateMLP, st_StateMLP)
#gmodel = Lux.StatefulLuxLayer(g_MLP, ps_trained.gMLP, st_gMLP)

# A smooth time grid for plotting the NN curve
t_plot = Float32.(collect(range(t_train[1], t_train[end], length=400)))

# NN predictions on the smooth grid (3×N)
ŷ_plot = smodel(permutedims(t_plot))  # 1×N input expected; returns 3×N

# Split true, NN, and noisy series
S_nn,   I_nn,   R_nn   = vec(ŷ_plot[1, :]), vec(ŷ_plot[2, :]), vec(ŷ_plot[3, :])
S_noisy, I_noisy, R_noisy = Y_train[1, :], Y_train[2, :], Y_train[3, :]

# Panel 1: S(t)
p1 = Plots.plot(t_plot, S_nn; label = "NN S(t)", linewidth = 2)
#Plots.plot!(p1, t_obs_array, S_true; label = "ODE S(t)", linestyle = :dash, linewidth = 2)
Plots.scatter!(p1, t_train, S_noisy; label = "data S", ms = 3, alpha = 0.7)
Plots.xlabel!(p1, "t")
Plots.ylabel!(p1, "S(t)")

# Panel 2: I(t)
p2 = Plots.plot(t_plot, I_nn; label = "NN I(t)", linewidth = 2)
# Plots.plot!(p2, t_obs_array, I_true; label = "ODE I(t)", linestyle = :dash, linewidth = 2)
Plots.scatter!(p2, t_train, I_noisy; label = "data I", ms = 3, alpha = 0.7)
Plots.xlabel!(p2, "t")
Plots.ylabel!(p2, "I(t)")

# Panel 3: R(t)
p3 = Plots.plot(t_plot, R_nn; label = "NN R(t)", linewidth = 2)
# Plots.plot!(p3, t_obs_array, R_true; label = "ODE R(t)", linestyle = :dash, linewidth = 2)
Plots.scatter!(p3, t_train, R_noisy; label = "data R", ms = 3, alpha = 0.7)
Plots.xlabel!(p3, "t")
Plots.ylabel!(p3, "R(t)")

# Combine panels
plt = Plots.plot(p1, p2, p3; layout = (3, 1), size = (600, 900))
display(plt)

#Plots.savefig(plt, "Results/2025-10-13_NNfitWithODELoss.svg")

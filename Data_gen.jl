# This file generates noisy data and saves it

using Random, Statistics
using DifferentialEquations


#--------True Physics--------

struct SIR_params
    β::Float32
    γ::Float32
end

function SIR_model!(du, u, p::SIR_params, t)
    S, I, R = u
    du[1] = -p.β * S * I
    du[2] = p.β * S * I - p.γ * I
    du[3] = p.γ * I
end

#--------Data Generation--------

function generate_data(;
                       seed = 42,
                       t_span = (0f0, 60f0),
                       n = 101,
                       u0 = (0.999f0, 0.001f0, 0.02f0),
                       ptrue = SIR_params(0.3f0, 0.1f0),
                       noise_pct = (0.2f0, 0.2f0, 0.2f0))

    rng = Random.MersenneTwister(seed)
    t = Float32.(collect(range(t_span[1], t_span[2], length=n)))
    prob = DifferentialEquations.ODEProblem(
        SIR_model!, Float32.([u0...]),
        (Float32.(t_span[1]), Float32.(t_span[2])),
        ptrue
    )
    sol = DifferentialEquations.solve(
        prob,
        DifferentialEquations.Tsit5();
        saveat = t
    )
    Yclean = Float32.(Array(sol)) # 3xN

    # Adding noise to the clean data

    noise_vec = [Float32.(noise_pct)...]
    noise = reshape(noise_vec, :, 1) .* Yclean .* randn(rng, size(Yclean)...)
    Y = clamp.(Yclean .+ noise, 0f0, Inf)
    Ystd = vec(Statistics.std(Y; dims = 2))
    
    Ystd = max.(Ystd, 1f-6)

    return (t=t, Y=Y, Yclean=Yclean, y0=Y[:,1], Ystd=Ystd, ptrue=ptrue, u0=Float32.([u0...]))
end

#--------Data Saving--------

using DataFrames, CSV

function save_dataset(path; format::Symbol=:csv, kwargs...)
    data = generate_data(; kwargs...)

    if format ==:csv
        base = endswith(path, ".csv") ? path[1:end-4] : path
        # main time-series table
        df = DataFrame(
            t        = data.t,
            S        = vec(data.Y[1, :]),
            I        = vec(data.Y[2, :]),
            R        = vec(data.Y[3, :]),
            S_clean  = vec(data.Yclean[1, :]),
            I_clean  = vec(data.Yclean[2, :]),
            R_clean  = vec(data.Yclean[3, :]),
        )

        CSV.write("$(base).csv", df)

    else
        error("Unknown format=$(format). Use :csv")

    end

    return nothing

end

#--------Visualizing Generated Dataset--------

function plot_dataset(data;
                      layout=(3,1), ms=3, lw=2, legend=:topright, title="SIR: noisy vs true")
    t      = data.t
    Y      = data.Y        # noisy: 3 x T
    Yclean = data.Yclean   # true:  3 x T

    plt = Plots.plot(layout=layout, legend=legend, size=(900, 800), title=title)

    # Keep consistent colors across scatter+line for each state
    labels = ("S", "I", "R")
    for (i, lbl) in enumerate(labels)
        Plots.scatter!(
            plt[i],
            t, vec(Y[i, :]); label="$lbl (noisy)", ms=ms,
            xlabel=i == 3 ? "t" : "", ylabel=lbl
        )
        Plots.plot!(
            plt[i],
            t, vec(Yclean[i, :]); label="$lbl (true)", lw=lw
        )
       # Plots.grid!(plt[i], true)
    end
    return plt
end

function plot_dataset_from_csv(basepath; kwargs...)
    path = endswith(basepath, ".csv") ? basepath : "$(basepath).csv"
    @assert isfile(path) "Could not find CSV at: $path"

    df = CSV.read(path, DataFrame)

    # Rebuild the minimal `data` NamedTuple shape used by plot_dataset
    t      = Float32.(df.t)
    Y      = permutedims(Float32.(hcat(df.S, df.I, df.R)))      # 3 x T
    Yclean = permutedims(Float32.(hcat(df.S_clean, df.I_clean, df.R_clean)))  # 3 x T

    data = (; t=t, Y=Y, Yclean=Yclean)

    return plot_dataset(data; kwargs...)
end

# --- Uncomment to quickly produce a dataset file ---

# save_dataset("Results/Noisy_SIR.csv";
#     seed=42, tspan=(0f0,60f0), n=101,
#     u0=(0.999f0,0.001f0,0f0),
#     SIR_params(0.3f0, 0.1f0),
#     noise_pct=(0.02f0, 0.05f0, 0.02f0)
# )


        
    

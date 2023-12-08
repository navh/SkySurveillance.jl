module SkySurveillance

using Base: Threads
using Dates: format, now
using Distributions: Normal, Uniform, pdf, logpdf, Categorical, Product, fill
using StatsBase: sample, Weights, loglikelihood, mean, entropy, std, var
using Flux: Zygote
using Flux
using JLD2: jldsave, load
using NNlib: softmax, logsoftmax
using POMDPTools:
    Deterministic,
    DisplaySimulator,
    HistoryRecorder,
    NothingUpdater,
    POMDPTools,
    RandomPolicy,
    RandomSolver,
    Sim,
    eachstep,
    stepthrough
using POMDPs: POMDP, POMDPs, Solver, Updater, discount, isterminal, reward, simulate, solve
using Parameters: @with_kw
using Plots: @animate, Plots, RGB, Shape, distinguishable_colors, heatmap!, mov, plot, plot!
using Random: AbstractRNG, Xoshiro, shuffle
using StaticArrays: @SMatrix, @SVector, SA, SMatrix, SVector
using TOML: TOML

PARAMS = TOML.parsefile(ARGS[1])
println("-------------------- begin params '$(ARGS[1])'")
TOML.print(PARAMS)
println("-------------------- end params '$(ARGS[1])'")

# if PARAMS["use_gpu"]
#     if PARAMS["gpu_type"] == "CUDA"
#         using CUDA
#         CUDA.allowscalar(true)
#     elseif PARAMS["gpu_type"] == "Metal"
#         using Metal
#         Metal.allowscalar(true)
#     end
# end
#

# include("utils/replay_buffer.jl")
# include("utils/config_parser.jl")
# include("utils/logger.jl")
# include("utils/multi_thread_env.jl")

include("Flat_POMDP/types.jl")
include("Flat_POMDP/flat_pomdp.jl")
include("Flat_POMDP/belief_pomdp.jl")
include("Flat_POMDP/updater.jl")
include("Flat_POMDP/solver_random.jl")
# include("Flat_POMDP/solver_sac.jl")
# include("Flat_POMDP/solver_ppo.jl")
# include("Flat_POMDP/belief_mdp.jl")
include("Flat_POMDP/visualizations.jl")

run_time = format(now(), "YYYYmmdd-HHMMSS-sss")
@info "Run: $(run_time)"

mkpath(PARAMS["log_path"])

rng = Xoshiro(PARAMS["seed"])
child_pomdp = FlatPOMDP(rng, DISCOUNT)
updater = MultiFilterUpdater(child_pomdp.rng)
pomdp = BeliefPOMDP(child_pomdp.rng, child_pomdp, updater)

#solver = RandomMultiFilter()
solver = RandomSolver()
policy = solve(solver, pomdp)

hr = HistoryRecorder(; max_steps=PARAMS["animation_steps"])
history = simulate(hr, pomdp, policy)

if PARAMS["render"]
    anim = @animate for step in eachstep(history)
        POMDPTools.render(pomdp, step)
    end
    mkpath(PARAMS["video_path"])
    mov(anim, "$(PARAMS["video_path"])$(run_time).mov"; loop=1)
end

rewards = [reward(pomdp, step.s, step.b) for step in eachstep(history)]

mkpath(PARAMS["log_path"])
open("$(PARAMS["log_path"])$(run_time).txt", "w") do f
    for i in rewards
        println(f, i)
    end
end

end
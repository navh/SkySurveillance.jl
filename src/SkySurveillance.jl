module SkySurveillance

using Distributions: Uniform
using Flux:
    Adam, Chain, Dense, Flux, LSTM, batchseq, chunk, cpu, gpu, logitcrossentropy, mse
using JLD2: jldsave, load
using POMDPPolicies: RandomPolicy
using POMDPTools:
    Deterministic,
    DisplaySimulator,
    HistoryRecorder,
    NothingUpdater,
    POMDPTools,
    RandomSolver,
    Sim,
    eachstep,
    stepthrough
using POMDPs: POMDP, POMDPs, Solver, Updater, discount, isterminal, reward, simulate, solve
using Parameters: @with_kw
using Plots: @animate, Plots, Shape, heatmap!, mov, plot, plot!
using Random: AbstractRNG, Xoshiro
using StaticArrays: @SMatrix, @SVector, SA, SMatrix, SVector
using TOML: TOML

PARAMS = TOML.parsefile(ARGS[1])
println("-------------------- begin params '$(ARGS[1])'")
TOML.print(PARAMS)
println("-------------------- end params '$(ARGS[1])'")

include("Flat_POMDP/flat_pomdp.jl")
include("Flat_POMDP/solver.jl")
include("Flat_POMDP/visualizations.jl")

rng = Xoshiro(PARAMS["seed"])

pomdp = FlatPOMDP(; rng=rng)

solver = FigOfflineSolver()
policy = solve(solver, pomdp)
#
# ds = DisplaySimulator(; max_steps=100, extra_final=false, spec="(s,a,o,b)")
# simulate(ds, pomdp, policy)
#
# hr = HistoryRecorder(; max_steps=50)
# history = simulate(hr, pomdp, policy)
# for step in eachstep(history)
#     #@show step.b
# end

anim = @animate for step in stepthrough(pomdp, policy; max_steps=1000)
    if PARAMS["render"]
        POMDPTools.render(pomdp, step)
    end
end
mov(anim, "./test2.mov"; loop=1)
#
# solver = QMDPSolver() # From QMDP
#
# policy = solve(solver, pomdp)
#
# belief_updater = updater(policy)

end

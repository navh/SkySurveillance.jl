module SkySurveillance

using Distributions: Uniform
using Flux: Adam, Chain, Dense, Flux, LSTM, batchseq, chunk, cpu, gpu, logitcrossentropy
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

include("Flat_POMDP/flat_pomdp.jl")
include("Flat_POMDP/solver.jl")
include("Flat_POMDP/visualizations.jl")

SEED::Int64 = 42
RENDER::Bool = true
WRITE_MODEL::Bool = true

# Solver Hyperparams
LEARNING_RATE::Float64 = 1e-2
STEPS_PER_SEQUENCE::Int = 500
TRAIN_SEQUENCES::Int = 500
TEST_SEQUENCES::Int = 5 # roughly 0.05 of train
EPOCHS::Int = 200
USE_GPU::Bool = false

# initialize the problem

rng = Xoshiro(SEED)

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
    if RENDER
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

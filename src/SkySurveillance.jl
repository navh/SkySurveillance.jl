module SkySurveillance

using Distributions: Uniform
using Flux: Chain, Dense, Flux, LSTM, batchseq, chunk, cpu, gpu, logitcrossentropy, Adam
using POMDPPolicies: RandomPolicy
using POMDPTools:
    Deterministic,
    HistoryRecorder,
    NothingUpdater,
    POMDPTools,
    RandomSolver,
    Sim,
    eachstep,
    stepthrough
using POMDPs: POMDP, POMDPs, Solver, Updater, discount, isterminal, reward, simulate, solve
using Parameters: @with_kw
using Plots: plot!, Plots, Shape
using Random: Xoshiro, AbstractRNG
using StaticArrays: @SMatrix, @SVector, SA, SMatrix, SVector
using JLD2: jldsave, load

include("Flat_POMDP/flat_pomdp.jl")
include("Flat_POMDP/solver.jl")

SEED::Int64 = 42
RENDER::Bool = true

# Solver Hyperparams
LEARNING_RATE::Float64 = 1e-2
STEPS_PER_SEQUENCE::Int = 500
STEPS_PER_SEQUENCE::Int = 50
SEQUENCES_PER_BATCH::Int = 50
TRAIN_SEQUENCES::Int = 100
TRAIN_SEQUENCES::Int = 10
TEST_SEQUENCES::Int = 50 # roughly 0.05 of train
TEST_SEQUENCES::Int = 5 # roughly 0.05 of train
EPOCHS::Int = 3
USE_GPU::Bool = false

# initialize the problem

rng = Xoshiro(SEED)

pomdp = FlatPOMDP(; rng=rng)

solver = FigOfflineSolver()
policy = solve(solver, pomdp)

hr = HistoryRecorder(; max_steps=50)
history = simulate(hr, pomdp, policy)
for step in eachstep(history)
    #@show step.b
end

# anim = @animate for step in stepthrough(pomdp, policy; max_steps=1000)
#     # if RENDER
#     POMDPTools.render(pomdp, step)
#     # end
# end
# mov(anim, "./test1.mov"; loop=1)
#
# solver = QMDPSolver() # From QMDP
#
# policy = solve(solver, pomdp)
#
# belief_updater = updater(policy)

end

module SkySurveillance

using CUDA
using CommonRLInterface: AbstractEnv
using Dates: format, now
using JLD2
using Distributions: Normal, Uniform
using Flux
using Flux: glorot_uniform, mse
using IntervalSets
using POMDPTools.BeliefUpdaters: NothingUpdater, PreviousObservationUpdater
using POMDPTools.Simulators: Sim, run_parallel, stepthrough
using POMDPTools:
    Deterministic, HistoryRecorder, POMDPTools, RandomPolicy, RandomSolver, eachstep
using POMDPs:
    POMDP, POMDPs, Policy, Solver, Updater, discount, isterminal, reward, simulate, solve
using Plots:
    @animate,
    Plots,
    RGB,
    Shape,
    distinguishable_colors,
    mp4,
    plot,
    plot!,
    pdf,
    savefig,
    pgfplotsx
using Random: AbstractRNG, Xoshiro
using StaticArrays: SVector
using Statistics: mean, var
using TOML: parse, parsefile

export FlatPOMDP, MultiFilterUpdater, BeliefPOMDP, RandomSolver, SimpleGreedySolver

include("Flat_POMDP/types.jl")
include("Flat_POMDP/flat_pomdp.jl")
include("Flat_POMDP/belief_pomdp.jl")
include("Flat_POMDP/updater.jl")
include("Flat_POMDP/solver_random.jl")
include("Flat_POMDP/solver_simple_net.jl")
include("Flat_POMDP/visualizations.jl")

end

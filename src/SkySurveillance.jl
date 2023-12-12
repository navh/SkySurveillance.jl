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
using Random: AbstractRNG, Xoshiro, default_rng
using StaticArrays: SVector
using Statistics: mean, var
using TOML: parse, parsefile

if isempty(ARGS)
    PARAMS = parse("""
seed = 4
render = true
animation_steps = 1000
video_path = "./animations/"
log_path = "./logs/"
figure_path = "./figures/"
model_path = "./models/"
number_of_targets = 10 
beamwidth_degrees = 10
radar_min_range_meters = 500
radar_max_range_meters = 500_000
dwell_time_seconds = 200e-3
target_velocity_max_meters_per_second = 700
n_particles = 100
""")
else
    PARAMS = parsefile(ARGS[1])
    @info "Parameters" PARAMS
end

include("Flat_POMDP/types.jl")
include("Flat_POMDP/flat_pomdp.jl")
include("Flat_POMDP/belief_pomdp.jl")
include("Flat_POMDP/updater.jl")
include("Flat_POMDP/solver_random.jl")
include("Flat_POMDP/solver_simple_net.jl")
include("Flat_POMDP/visualizations.jl")

#### Run Experiment
include("experiments/generate_comparison.jl")

end
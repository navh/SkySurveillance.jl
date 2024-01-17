using CUDA: NVCL_CTX_SCHED_SPIN
include("../src/SkySurveillance.jl")
using .SkySurveillance:
    FlatPOMDP, MultiFilterUpdater, BeliefPOMDP, RandomSolver, SimpleGreedySolver

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
    println("ARGS empty")
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
    println("parse ARGS")
    PARAMS = parsefile(ARGS[1])
    @info "Parameters" PARAMS
end

# N_EPOCHS = 1
# N_TRAIN_EPISODES = 100
# N_TEST_EPISODES = 10
# N_SKIP_FIRST_STEPS = 100
# N_STEPS_PER_EPISODE = 500
# MONTE_CARLO_ROLLOUTS = 100

N_EPOCHS = 1
N_TRAIN_EPISODES = 10
N_TEST_EPISODES = 10
N_SKIP_FIRST_STEPS = 100
N_STEPS_PER_EPISODE = 100
MONTE_CARLO_ROLLOUTS = 10

run_time = format(now(), "YYYYmmdd-HHMMSS-sss")
@info "Run: $(run_time)"

mkpath(PARAMS["log_path"])

rng = Xoshiro(PARAMS["seed"])

# mutable struct FlatPOMDP <: POMDP{FlatState,FlatAction,FlatObservation} # POMDP{State, Action, Observation}
#     rng::AbstractRNG
#     discount::Float64
#     number_of_targets::Int64
#     beamwidth_rad::Float64
#     radar_min_range_meters::Int64
#     radar_max_range_meters::Int64
#     n_particles::Int64
#     xy_min_meters::Float64
#     xy_max_meters::Float64
#     dwell_time_seconds::Float64 # ∈ [10ms,40ms] # from Jack
#     target_velocity_max_meters_per_second::Float64 # rounded up f-22 top speed is 700m/s
#     # target_reappearing_distribution::Union{Uniform,Deterministic{Int64}} # should be some generic 'distribution' type
#     target_reappearing_distribution::Deterministic{Int64} # should be some generic 'distribution' type
# end

child_pomdp = FlatPOMDP(;
    rng=rng,
    discount=0.95, # was 1.0
    number_of_targets=PARAMS["number_of_targets"],
    beamwidth_rad=PARAMS["beamwidth_degrees"] * π / 180,
    radar_min_range_meters=PARAMS["radar_min_range_meters"],
    radar_max_range_meters=PARAMS["radar_max_range_meters"],
    n_particles=PARAMS["n_particles"],
    xy_min_meters=-1 * PARAMS["radar_max_range_meters"],
    xy_max_meters=PARAMS["radar_max_range_meters"],
    dwell_time_seconds=PARAMS["dwell_time_seconds"],
    target_velocity_max_meters_per_second=PARAMS["target_velocity_max_meters_per_second"],
    target_reappearing_distribution=Deterministic(0), # was Uniform(-50, 0)
)
u = MultiFilterUpdater(
    child_pomdp.rng, child_pomdp.dwell_time_seconds, child_pomdp.n_particles
)
pomdp = BeliefPOMDP(child_pomdp.rng, child_pomdp, u)

random_solver = RandomSolver()
random_policy = solve(random_solver, pomdp)
learned_solver = SimpleGreedySolver(;
    pomdp=pomdp,
    n_epochs=N_EPOCHS,
    n_train_episodes=N_TRAIN_EPISODES,
    n_test_episodes=N_TEST_EPISODES,
    n_skip_first_steps=N_SKIP_FIRST_STEPS,
    n_steps_per_episode=N_STEPS_PER_EPISODE,
    monte_carlo_rollouts=MONTE_CARLO_ROLLOUTS,
)
learned_policy = solve(learned_solver, pomdp)

# TODO: does recycling the HistoryRecorder work?
hr = HistoryRecorder(; max_steps=PARAMS["animation_steps"])

function plot_everything()
    all_learned = []
    all_random = []
    for _ in 1:25
        learned_history = simulate(hr, pomdp, learned_policy)
        random_history = simulate(hr, pomdp, random_policy)
        push!(
            all_learned,
            [reward(pomdp, step.s, step.b) for step in eachstep(learned_history)],
        )
        push!(
            all_random, [reward(pomdp, step.s, step.b) for step in eachstep(random_history)]
        )
    end

    learned_rewards = mean(all_learned)
    random_rewards = mean(all_random)

    plt = plot(;
        xlabel="Step",
        ylabel="Belief-Reward",
        size=(500, 350),
        titlefont=("times"),
        legendfont=("times"),
        # tickfont=("times"),
        guidefont=("times"),
    )
    for abc in all_random
        plt = plot!(plt, abc; label="", color=:blue, alpha=0.15)
    end
    for abc in all_learned
        plt = plot!(plt, abc; label="", color=:red, alpha=0.15)
    end
    #
    plt = plot!(plt, random_rewards; label="Mean Random Policy", color=:blue)
    plt = plot!(plt, learned_rewards; label="Mean Learned Policy", color=:red)
    #
    return savefig(plt, "pdfs/comparison_$(run_time).pdf")
end
plot_everything()

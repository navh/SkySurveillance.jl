include("../src/SkySurveillance.jl")
using .SkySurveillance:
    FlatPOMDP, MultiFilterUpdater, BeliefPOMDP, RandomSolver, SimpleGreedySolver

using Dates: format, now
using JLD2
using POMDPTools: POMDPTools
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
using Random: Xoshiro
using TOML: parse, parsefile

run_time = format(now(), "YYYYmmdd-HHMMSS-sss")
@info "Run: $(run_time)"

PARAMS = Dict(
    "seed" => 4,
    "render" => true,
    "animation_steps" => 1000,
    "video_path" => "./animations/",
    "log_path" => "./logs/",
    "figure_path " => "./figures/",
    "model_path" => "./models/",
    "number_of_targets" => 10,
    "beamwidth_degrees" => 10,
    "radar_min_range_meters" => 500,
    "radar_max_range_meters" => 500_000,
    "dwell_time_seconds" => 200e-3,
    "target_velocity_max_meters_per_second" => 700,
    "n_particles" => 100,
    "n_epochs" => 1,
    "n_train_episodes" => 10,
    "n_test_episodes" => 10,
    "n_skip_first_steps" => 100,
    "n_steps_per_episode" => 500,
    "monte_carlo_rollouts" => 100,
)

if isempty(ARGS)
    @info "ARGS empty"
else
    @info "parsing ARGS"
    parsed_params = parsefile(ARGS[1])
    for (k, v) in parsed_params
        PARAMS[k] = v
    end
end

@info "Parameters" PARAMS
mkpath(PARAMS["log_path"])

rng = Xoshiro(PARAMS["seed"])

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
    n_epochs=PARAMS["n_epochs"],
    n_train_episodes=PARAMS["n_train_episodes"],
    n_test_episodes=PARAMS["n_test_episodes"],
    n_skip_first_steps=PARAMS["n_skip_first_steps"],
    n_steps_per_episode=PARAMS["n_steps_per_episode"],
    monte_carlo_rollouts=PARAMS["monte_carlo_rollouts"],
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

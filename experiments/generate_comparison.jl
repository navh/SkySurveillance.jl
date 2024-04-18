include("experiment_utils.jl")

@info "Define solvers"
solver_random = SequentialSolver()
solver_sequential = SequentialSolver()
solver_highest_variance = HighestVarianceSolver(0.4)
# solver_single_sweep = SingleSweepSolver()

# learned_solver = SimpleGreedySolver(;
#     pomdp=pomdp,
#     n_epochs=PARAMS["n_epochs"],
#     n_train_episodes=PARAMS["n_train_episodes"],
#     n_test_episodes=PARAMS["n_test_episodes"],
#     n_skip_first_steps=PARAMS["n_skip_first_steps"],
#     n_steps_per_episode=PARAMS["n_steps_per_episode"],
#     monte_carlo_rollouts=PARAMS["monte_carlo_rollouts"],
# )

@info "Define policies"
policy_random = solve(solver_random, bpomdp)
policy_sequential = solve(solver_sequential, bpomdp)
policy_highest_variance = solve(solver_highest_variance, bpomdp)
# policy_single_sweep = solve(solver_single_sweep, pomdp)

skip_steps = 300
hr = HistoryRecorder(; max_steps=PARAMS["animation_steps"] + skip_steps)
# history = simulate(hr, bpomdp, policy, up) # specify updater for BetaZero

# function eachstep_skipped(history, skip_steps)
#     history = [step for step in eachstep(history)]
#     return history[skip_steps:end]
# end

@info "Generate plots"
function plot_everything()
    all_random = []
    all_sequential = []
    all_highest_variance = []
    for _ in 1:1000
        history_random = simulate(hr, bpomdp, policy_random)
        history_sequential = simulate(hr, bpomdp, policy_sequential)
        history_highest_variance = simulate(hr, bpomdp, policy_highest_variance)
        push!(
            all_random,
            [reward(bpomdp, step.s, step.b) for step in eachstep(history_random)],
        )
        push!(
            all_sequential,
            [reward(bpomdp, step.s, step.b) for step in eachstep(history_sequential)],
        )
        push!(
            all_highest_variance,
            [reward(bpomdp, step.s, step.b) for step in eachstep(history_highest_variance)],
        )
    end

    all_random = [h[skip_steps:end] for h in all_random]
    all_sequential = [h[skip_steps:end] for h in all_sequential]
    all_highest_variance = [h[skip_steps:end] for h in all_highest_variance]

    plt = plot(;
        xlabel="Step",
        ylabel="Belief-Reward",
        #size=(3.5 * 72, 2.5 * 72),
        size=(16 / 3 * 72, 9 / 3 * 72),
        # fontfamily="Times New Roman",
        fontfamily="Helvetica",
        grid=false,
        fg_legend=:transparent,
        framestyle=:origin,
    )

    # for reward in all_random
    #     plt = plot!(plt, reward; label="", color=:black, alpha=0.1)
    # end
    # for reward in all_sequential
    #     plt = plot!(plt, reward; label="", color=:blue, alpha=0.1)
    # end
    # for reward in all_highest_variance
    #     plt = plot!(plt, reward; label="", color=:green, alpha=0.1)
    # end
    # 

    # plt = plot!(plt, mean(all_random); label="Random", color=:black)
    # plt = plot!(plt, mean(all_sequential); label="Sequential", color=:blue)
    # plt = plot!(plt, mean(all_highest_variance); label="Heuristic", color=:green)
    plt = plot!(plt, mean(all_random); label="Random")
    plt = plot!(plt, mean(all_sequential); label="Sequential")
    plt = plot!(plt, mean(all_highest_variance); label="Heuristic")

    @info "Writing figure"
    fig_path = dir_paths.figure_dir * "comparison.pdf"
    @info fig_path
    return savefig(plt, fig_path)
end
plot_everything()

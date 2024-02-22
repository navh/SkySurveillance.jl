include("experiment_utils.jl")

@info "Define solvers"
solver_random = SequentialSolver()
solver_sequential = SequentialSolver()
solver_highest_variance = HighestVarianceSolver()
solver_single_sweep = SingleSweepSolver()

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
policy_random = solve(solver_random, pomdp)
policy_sequential = solve(solver_sequential, pomdp)
policy_highest_variance = solve(solver_highest_variance, pomdp)
policy_single_sweep = solve(solver_single_sweep, pomdp)

hr = HistoryRecorder(; max_steps=PARAMS["animation_steps"])

@info "Generate plots"
function plot_everything()
    all_random = []
    all_sequential = []
    all_highest_variance = []
    all_single_sweep = []
    for _ in 1:10
        history_random = simulate(hr, pomdp, policy_random)
        history_sequential = simulate(hr, pomdp, policy_sequential)
        history_highest_variance = simulate(hr, pomdp, policy_highest_variance)
        history_single_sweep = simulate(hr, pomdp, policy_single_sweep)
        push!(
            all_random, [reward(pomdp, step.s, step.b) for step in eachstep(history_random)]
        )
        push!(
            all_sequential,
            [reward(pomdp, step.s, step.b) for step in eachstep(history_sequential)],
        )
        push!(
            all_highest_variance,
            [reward(pomdp, step.s, step.b) for step in eachstep(history_highest_variance)],
        )
        push!(
            all_single_sweep,
            [reward(pomdp, step.s, step.b) for step in eachstep(history_single_sweep)],
        )
    end
    plt = plot(;
        xlabel="Step",
        ylabel="Belief-Reward",
        size=(500, 350),
        titlefont=("STIXTwoText"),
        legendfont=("STIXTwoText"),
        tickfont=("STIXTwoText"),
        guidefont=("STIXTwoText"),
    )
    for reward in all_single_sweep
        plt = plot!(plt, reward; label="", alpha=1)
    end

    # for reward in all_sequential
    #     plt = plot!(plt, reward; label="", alpha=1)
    # end
    # for reward in all_highest_variance
    #     plt = plot!(plt, reward; label="", color=:green, alpha=0.1)
    # end
    # for reward in all_random
    #     plt = plot!(plt, reward; label="", color=:red, alpha=0.1)
    # end

    # plt = plot!(plt, mean(all_sequential); label="Mean Sequential Policy", color=:blue)
    # plt = plot!(
    #     plt, mean(all_highest_variance); label="Mean Highest Variance Policy", color=:green
    # )
    # plt = plot!(plt, mean(all_random); label="Mean Learned Policy", color=:red)

    @info "Writing figure"
    fig_path = dir_paths.figure_dir * "comparison.pdf"
    @info fig_path
    return savefig(plt, fig_path)
end
plot_everything()

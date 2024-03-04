include("experiment_utils.jl")

@info "Define solvers"
solver_random = SequentialSolver()
solver_sequential = SequentialSolver()
solver_highest_variance = HighestVarianceSolver()
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

hr = HistoryRecorder(; max_steps=PARAMS["animation_steps"])

@info "Generate plots"
function plot_everything()
    all_random = []
    all_sequential = []
    all_highest_variance = []
    all_highest_varianc = []
    for _ in 1:1000
        history_random = simulate(hr, bpomdp, policy_random)
        history_sequential = simulate(hr, bpomdp, policy_sequential)
        history_highest_variance = simulate(hr, bpomdp, policy_highest_variance)
        # history_single_sweep = simulate(hr, pomdp, policy_single_sweep)
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
    plt = plot(;
        xlabel="Step",
        ylabel="Belief-Reward",
        size=(3.5 * 72, 2.5 * 72),
        # titlefont=("STIXTwoText"),
        fontfamily=("Times Roman"),
        # titlefont=("newtx"),
        # legendfont=("newtx"),
        # tickfont=("newtx"),
        # guidefont=("newtx"),
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

    plt = plot!(plt, mean(all_random); label="Random", color=:black, linestyle=":")
    plt = plot!(plt, mean(all_sequential); label="Sequential", color=:blue, linestyle="-")
    plt = plot!(plt, mean(all_highest_variance); label="ϵ-Greedy", color=:green)

    @info "Writing figure"
    fig_path = dir_paths.figure_dir * "comparison.pdf"
    @info fig_path
    return savefig(plt, fig_path)
end
plot_everything()

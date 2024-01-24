include("experiment_utils.jl")

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
    return savefig(plt, dir_paths.figure_dir * "comparison.pdf")
end
plot_everything()


run_time = format(now(), "YYYYmmdd-HHMMSS-sss")
@info "Run: $(run_time)"

mkpath(PARAMS["log_path"])

rng = Xoshiro(PARAMS["seed"])
child_pomdp = FlatPOMDP(rng, DISCOUNT)
u = MultiFilterUpdater(child_pomdp.rng)
pomdp = BeliefPOMDP(child_pomdp.rng, child_pomdp, u)
#solver = RandomMultiFilter()
random_solver = RandomSolver()
random_policy = solve(random_solver, pomdp)
learned_solver = SimpleGreedySolver()
learned_policy = solve(learned_solver, pomdp)
# learned_solver = SequentialSolver()

function plot_everything()
    rng = Xoshiro(PARAMS["seed"])
    child_pomdp = FlatPOMDP(rng, DISCOUNT)
    pomdp = BeliefPOMDP(child_pomdp.rng, child_pomdp, u)
    hr = HistoryRecorder(; max_steps=PARAMS["animation_steps"])

    child_pomdp = FlatPOMDP(rng, DISCOUNT)
    pomdp = BeliefPOMDP(child_pomdp.rng, child_pomdp, u)
    hr = HistoryRecorder(; max_steps=PARAMS["animation_steps"])

    all_learned = []
    all_random = []
    for i in 1:25
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
    #
    # mkpath(PARAMS["log_path"])
    # open("$(PARAMS["log_path"])$(run_time)-learned.txt", "w") do f
    #     for i in learned_rewards
    #         println(f, i)
    #     end
    # end
    # open("$(PARAMS["log_path"])$(run_time)-random.txt", "w") do f
    #     for i in random_rewards
    #         println(f, i)
    #     end
    # end
    #
    #pgfplotsx()
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

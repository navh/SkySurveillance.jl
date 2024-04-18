include("experiment_utils.jl") # Sets up the BMDP

# solver = SimpleGreedySolver(;
#     pomdp=pomdp,
#     n_epochs=PARAMS["n_epochs"],
#     n_train_episodes=PARAMS["n_train_episodes"],
#     n_test_episodes=PARAMS["n_test_episodes"],
#     n_skip_first_steps=PARAMS["n_skip_first_steps"],
#     n_steps_per_episode=PARAMS["n_steps_per_episode"],
#     monte_carlo_rollouts=PARAMS["monte_carlo_rollouts"],
# )
# solver = SingleSweepSolver()
# solver = SequentialSolver()
solver = HighestVarianceSolver(0.5)
#
policy = solve(solver, bpomdp)

# sixhundred =  "./out/20240229-102551-600/models/"

# runpath = "./out/20240229-104117-934/models/"
# policy = load_policy(runpath * "policy.bson")
# solver = load_solver(runpath * "solver.bson")

skip_steps = 200
hr = HistoryRecorder(; max_steps=PARAMS["animation_steps"] + skip_steps)
history = simulate(hr, bpomdp, policy)
history = [step for step in eachstep(history)]
history = history[skip_steps:end]
# history = simulate(hr, bpomdp, policy, up) # specify updater for BetaZero

# for step in history
#     println(step.s.underlying_state.azimuth_recency[30])
# end

@info "beginning rendering"
# anim = @animate for step in eachstep(history)
anim = @animate for step in history
    POMDPTools.render(bpomdp, step)
    # POMDPTools.render(pomdp, step) # oh just go ahead and create a different BetaZero animation at this point 
end
@info "writing animation"
mp4(anim, dir_paths.animation_dir * "a.mp4"; loop=1, fps=6)

@info "writing belief-reward history"
plt = plot(;
    xlabel="Step",
    ylabel="Belief-Reward",
    # There's 72 points in an inch
    # In theory an ieee col is 3.5 inches wide
    # size=(3.5 * 72, 2.5 * 72),
    # bigger for ppt
    size=(4 * 72, 3 * 72),
    # fontfamily="Times New Roman",
    # fontfamily="Arial",
    fontfamily="Helvetica",
    # fontfamily="Computer Modern",
    grid=false,
    fg_legend=:transparent,
)
rewards = [step.r for step in history]
track_rewards = [
    score_tracking(bpomdp.underlying_pomdp, step.s.underlying_state, step.s.belief_state)
    for step in history
]
recency_rewards = [score_search(step.s.belief_state) for step in history]

accuracy = []
igain = []
for i in 1:(length(rewards) - 1)
    push!(accuracy, rewards[i + 1])
    push!(igain, max(0, rewards[i + 1] - rewards[i]))
end
# plot!(plt, accuracy; label="Accuracy")
# plot!(plt, igain; label="Information Gain")
plot!(plt, accuracy; label="Quality")
plot!(plt, igain; label="Information")
plot!(plt, track_rewards; label="Track")
plot!(plt, recency_rewards; label="Search", legend=:best)
# plot!(plt, igain; label="Information Gain")
# plot!(plt, igain; label="Information Gain", legend=:inside)
# plot!(plt, igain; label="Information Gain", legend=:right)
fig_path = dir_paths.figure_dir * "single_hit.pdf"
@info fig_path
savefig(plt, fig_path)

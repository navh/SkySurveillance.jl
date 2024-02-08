include("experiment_utils.jl") # Sets up the BMDP
using Plots: @animate, Plots, mp4

# solver = SimpleGreedySolver(;
#     pomdp=pomdp,
#     n_epochs=PARAMS["n_epochs"],
#     n_train_episodes=PARAMS["n_train_episodes"],
#     n_test_episodes=PARAMS["n_test_episodes"],
#     n_skip_first_steps=PARAMS["n_skip_first_steps"],
#     n_steps_per_episode=PARAMS["n_steps_per_episode"],
#     monte_carlo_rollouts=PARAMS["monte_carlo_rollouts"],
# )
solver = SequentialSolver()

policy = solve(solver, pomdp)

hr = HistoryRecorder(; max_steps=PARAMS["animation_steps"])
history = simulate(hr, pomdp, policy)

@info "beginning rendering"
anim = @animate for step in eachstep(history)
    POMDPTools.render(pomdp, step)
end
@info "writing animation"
mp4(anim, dir_paths.animation_dir * "a.mp4"; loop=1)

@info "sorting rewards"
rewards = [reward(pomdp, step.s, step.b) for step in eachstep(history)]
@info "writing rewards"
open(dir_paths.log_dir * "rewards.txt", "w") do f
    for i in rewards
        println(f, i)
    end
end

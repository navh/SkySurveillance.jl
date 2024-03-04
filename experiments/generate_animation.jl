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
solver = SequentialSolver()
# solver = SingleSweepSolver()
# solver = HighestVarianceSolver()
# solver = HighestVarianceSolve()
#
policy = solve(solver, bpomdp)

# sixhundred =  "./out/20240229-102551-600/models/"

# runpath = "./out/20240229-104117-934/models/"
# policy = load_policy(runpath * "policy.bson")
# solver = load_solver(runpath * "solver.bson")

hr = HistoryRecorder(; max_steps=PARAMS["animation_steps"])
history = simulate(hr, bpomdp, policy)
# history = simulate(hr, bpomdp, policy, up) # specify updater for BetaZero

@info "beginning rendering"
anim = @animate for step in eachstep(history)
    POMDPTools.render(bpomdp, step)
    # POMDPTools.render(pomdp, step) # oh just go ahead and create a different BetaZero animation at this point 
end
@info "writing animation"
mp4(anim, dir_paths.animation_dir * "a.mp4"; loop=1)

include("experiment_utils.jl") # Sets up the BMDP

# runpath = "./out/20240229-102551-600/models/"

runpath = "./out/20240229-104117-934/models/"
policy = load_policy(runpath * "policy.bson")
solver = load_solver(runpath * "solver.bson")

hr = HistoryRecorder(; max_steps=PARAMS["animation_steps"])
history = simulate(hr, pomdp, policy, up) # specify updater, use flatPOMDP for BetaZero

@info "beginning rendering"
anim = @animate for step in eachstep(history)
    POMDPTools.render(pomdp, step)
end
@info "writing animation"
mp4(anim, dir_paths.animation_dir * "learned.mp4"; loop=1)

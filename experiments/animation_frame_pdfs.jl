include("experiment_utils.jl") # Sets up the BMDP

solver = SequentialSolver()
# solver = SingleSweepSolver()
# solver = HighestVarianceSolver()

policy = solve(solver, bpomdp)

hr = HistoryRecorder(; max_steps=PARAMS["animation_steps"])
history = simulate(hr, bpomdp, policy)

@info "beginning rendering"
for step in eachstep(history)
    pdf(POMDPTools.render(bpomdp, step), dir_paths.figure_dir * "frame-$(step.t).pdf")
end
@info dir_paths.figure_dir
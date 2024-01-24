include("experiment_utils.jl")

#solver = RandomMultiFilter()
solver = RandomSolver()
#solver = SimpleGreedySolver()
policy = solve(solver, pomdp)

hr = HistoryRecorder(; max_steps=PARAMS["animation_steps"])
history = simulate(hr, pomdp, policy)

for step in eachstep(history)
    pdf(POMDPTools.render(pomdp, step), dir_paths.figure_dir * "frame-$(step.t).pdf")
end

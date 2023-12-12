run_time = format(now(), "YYYYmmdd-HHMMSS-sss")
@info "Run: $(run_time)"

mkpath(PARAMS["log_path"])

rng = Xoshiro(PARAMS["seed"])
child_pomdp = FlatPOMDP(rng, DISCOUNT)
u = MultiFilterUpdater(child_pomdp.rng)
pomdp = BeliefPOMDP(child_pomdp.rng, child_pomdp, u)
#solver = RandomMultiFilter()
solver = RandomSolver()
#solver = SimpleGreedySolver()
policy = solve(solver, pomdp)

hr = HistoryRecorder(; max_steps=PARAMS["animation_steps"])
history = simulate(hr, pomdp, policy)

mkpath("./pdfs/")
for step in eachstep(history)
    pdf(POMDPTools.render(pomdp, step), "$("./pdfs/")$(step.t).pdf")
end
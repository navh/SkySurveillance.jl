run_time = format(now(), "YYYYmmdd-HHMMSS-sss")
@info "Run: $(run_time)"

mkpath(PARAMS["log_path"])

rng = Xoshiro(PARAMS["seed"])
child_pomdp = FlatPOMDP(rng, DISCOUNT)
u = MultiFilterUpdater(child_pomdp.rng)
pomdp = BeliefPOMDP(child_pomdp.rng, child_pomdp, u)
#solver = RandomMultiFilter()
#solver = RandomSolver()
solver = SimpleGreedySolver()
policy = solve(solver, pomdp)

hr = HistoryRecorder(; max_steps=PARAMS["animation_steps"])
history = simulate(hr, pomdp, policy)
#
# for step in eachstep(history)
#     pdf(POMDPTools.render(pomdp, step), "$("./pdfs/")$(step.t).pdf")
# end
#
if PARAMS["render"]
    anim = @animate for step in eachstep(history)
        POMDPTools.render(pomdp, step)
    end
    mkpath(PARAMS["video_path"])
    mp4(anim, "$(PARAMS["video_path"])$(run_time).mp4"; loop=1)
end

rewards = [reward(pomdp, step.s, step.b) for step in eachstep(history)]

mkpath(PARAMS["log_path"])
open("$(PARAMS["log_path"])$(run_time).txt", "w") do f
    for i in rewards
        println(f, i)
    end
end

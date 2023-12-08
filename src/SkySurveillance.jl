module SkySurveillance

using Dates: format, now
using Distributions: Normal, Uniform
using StatsBase: mean, var
using POMDPTools:
    Deterministic, HistoryRecorder, POMDPTools, RandomPolicy, RandomSolver, eachstep
using POMDPs: POMDP, POMDPs, Solver, Updater, discount, isterminal, reward, simulate, solve
using Plots: @animate, Plots, RGB, Shape, distinguishable_colors, mov, plot, plot!
using Random: AbstractRNG, Xoshiro
using StaticArrays: SVector
using TOML: parse, parsefile

if isempty(ARGS)
    PARAMS = parse("""
seed = 42
render = true
animation_steps = 1000
video_path = "./animations/"
log_path = "./logs/"
figure_path = "./figures/"
number_of_targets = 10
beamwidth_degrees = 10
radar_min_range_meters = 500
radar_max_range_meters = 500_000
dwell_time_seconds = 200e-3
target_velocity_max_meters_per_second = 700
n_particles = 100
""")
else
    PARAMS = parsefile(ARGS[1])
    @info "Parameters" PARAMS
end

include("Flat_POMDP/types.jl")
include("Flat_POMDP/flat_pomdp.jl")
include("Flat_POMDP/belief_pomdp.jl")
include("Flat_POMDP/updater.jl")
include("Flat_POMDP/solver_random.jl")
include("Flat_POMDP/visualizations.jl")

run_time = format(now(), "YYYYmmdd-HHMMSS-sss")
@info "Run: $(run_time)"

mkpath(PARAMS["log_path"])

rng = Xoshiro(PARAMS["seed"])
child_pomdp = FlatPOMDP(rng, DISCOUNT)
updater = MultiFilterUpdater(child_pomdp.rng)
pomdp = BeliefPOMDP(child_pomdp.rng, child_pomdp, updater)

#solver = RandomMultiFilter()
solver = RandomSolver()
policy = solve(solver, pomdp)

hr = HistoryRecorder(; max_steps=PARAMS["animation_steps"])
history = simulate(hr, pomdp, policy)

if PARAMS["render"]
    anim = @animate for step in eachstep(history)
        POMDPTools.render(pomdp, step)
    end
    mkpath(PARAMS["video_path"])
    mov(anim, "$(PARAMS["video_path"])$(run_time).mov"; loop=1)
end

rewards = [reward(pomdp, step.s, step.b) for step in eachstep(history)]

mkpath(PARAMS["log_path"])
open("$(PARAMS["log_path"])$(run_time).txt", "w") do f
    for i in rewards
        println(f, i)
    end
end

end
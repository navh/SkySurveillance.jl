include("../src/SkySurveillance.jl")
using .SkySurveillance:
    BeliefPOMDP, FlatPOMDP, MultiFilterUpdater, RandomSolver, SimpleGreedySolver
using Dates: format, now
using POMDPs: solve, simulate, reward, mean
using POMDPTools: Deterministic, POMDPTools, HistoryRecorder, eachstep
using Plots: plot, plot!, savefig, @animate, mp4, Plots, pdf
using Random: Xoshiro
using TOML

run_time = format(now(), "YYYYmmdd-HHMMSS-sss")
@info "Run: $(run_time)"

PARAMS = Dict(
    "seed" => 0,
    "render" => true,
    "animation_steps" => 1000,
    "output_path" => "./out",
    "number_of_targets" => 10,
    "beamwidth_degrees" => 10,
    "radar_min_range_meters" => 500,
    "radar_max_range_meters" => 500_000,
    "dwell_time_seconds" => 200e-3,
    "target_velocity_max_meters_per_second" => 700,
    "n_particles" => 100,
    "n_epochs" => 1,
    "n_train_episodes" => 10,
    "n_test_episodes" => 10,
    "n_skip_first_steps" => 100,
    "n_steps_per_episode" => 500,
    "monte_carlo_rollouts" => 100,
    "discount" => 0.95,
)
if isempty(ARGS)
    @info "ARGS empty"
else
    @info "parsing ARGS"
    parsed_params = TOML.parsefile(ARGS[1])
    for (key, value) in parsed_params
        PARAMS[key] = value
    end
end

output_dir = "$(PARAMS["output_path"])/$(run_time)/"
dir_paths = (
    output_dir=output_dir,
    animation_dir=output_dir * "animations/",
    log_dir=output_dir * "logs/",
    figure_dir=output_dir * "figures/",
    model_dir=output_dir * "models/",
)
for dir_path in dir_paths
    mkpath(dir_path)
end

@info "Writing params.toml"
open(dir_paths.output_dir * "params.toml", "w") do io
    TOML.print(io, PARAMS; sorted=true)
end

rng = Xoshiro(PARAMS["seed"])

child_pomdp = FlatPOMDP(;
    rng=rng,
    discount=PARAMS["discount"],
    number_of_targets=PARAMS["number_of_targets"],
    beamwidth_rad=PARAMS["beamwidth_degrees"] * π / 180,
    radar_min_range_meters=PARAMS["radar_min_range_meters"],
    radar_max_range_meters=PARAMS["radar_max_range_meters"],
    n_particles=PARAMS["n_particles"],
    xy_min_meters=-1 * PARAMS["radar_max_range_meters"],
    xy_max_meters=PARAMS["radar_max_range_meters"],
    dwell_time_seconds=PARAMS["dwell_time_seconds"],
    target_velocity_max_meters_per_second=PARAMS["target_velocity_max_meters_per_second"],
    target_reappearing_distribution=Deterministic(0), # was Uniform(-50, 0)
)
u = MultiFilterUpdater(
    child_pomdp.rng, child_pomdp.dwell_time_seconds, child_pomdp.n_particles
)
pomdp = BeliefPOMDP(child_pomdp.rng, child_pomdp, u)

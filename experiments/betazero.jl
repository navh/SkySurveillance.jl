include("../src/SkySurveillance.jl")
using .SkySurveillance
using BetaZero
using Dates: format, now
using Distributions: Uniform
using POMDPTools: Deterministic, HistoryRecorder, POMDPTools, eachstep
using POMDPs: mean, reward, simulate, solve
using Plots: @animate, Plots, mp4, pdf, plot, plot!, savefig
using Random: Xoshiro
using TOML

run_time = format(now(), "YYYYmmdd-HHMMSS-sss")
@info "Run: $(run_time)"

PARAMS = Dict(
    "seed" => 1,
    "render" => true,
    "animation_steps" => 1200,
    "output_path" => "./out",
    "number_of_targets" => 20,
    "beamwidth_degrees" => 360 / 20, # 20 slices
    "radar_min_range_meters" => 500,
    "radar_max_range_meters" => 500_000,
    "dwell_time_seconds" => 200e-3,
    "target_velocity_max_meters_per_second" => 700,
    "n_particles" => 200,
    "maximum_filter_variance" => 1e9,
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

pomdp = FlatPOMDP(;
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
    target_reappearing_distribution=Uniform(-500, 0),
)

up = MultiFilterUpdater(
    pomdp.rng,
    pomdp.dwell_time_seconds,
    pomdp.n_particles,
    pomdp.radar_max_range_meters,
    PARAMS["maximum_filter_variance"],
)
function POMDPs.isterminal(
    bmdp::BeliefMDP{FlatPOMDP,MultiFilterUpdater,Vector{SingleFilter},Number},
    b::Vector{SingleFilter},
)
    return false
end
# up = BootstrapFilter(pomdp, 500)
#
MAX_INPUT_TARGETS = 20 # note: gets doubled, must be >= max particles

function filter_mean_θ(filter::SingleFilter)
    # copied from updater.jl, should not be defined in experiment
    return mean([atan(particle.y, particle.x) for particle in filter.particles])
end
function filter_variance(filter::SingleFilter)
    # copied from updater.jl, should not be defined in experiment
    return (var([p.x for p in filter.particles]) + var([p.y for p in filter.particles])) / 2 # just use mean? 
end

function BetaZero.input_representation(belief::Array{SingleFilter})
    # Function to get belief representation as input to neural network.
    # μ, σ = mean_and_std(s.y for s in particles(b))
    θs_and_variances = sort([
        (filter_mean_θ(filter), filter_variance(filter)) for filter in belief
    ])
    # θs_and_variances = (
    #     (filter_mean_θ(filter), filter_variance(filter)) for filter in belief
    # )
    #
    return Vector{Float32}(
        vcat(
            [θ for (θ, var) in θs_and_variances],
            zeros(Float32, MAX_INPUT_TARGETS - length(θs_and_variances)),
            [var for (θ, var) in θs_and_variances],
            zeros(Float32, MAX_INPUT_TARGETS - length(θs_and_variances)),
        ),
    )
end

function random_target_from_filter(rng::AbstractRNG, filter::SingleFilter)
    random_particle = rand(rng, filter.particles)
    return Target(
        filter.id,
        1.0,
        random_particle.x,
        random_particle.y,
        random_particle.ẋ,
        random_particle.ẏ,
    )
end

function POMDPs.gen(bmdp::BeliefMDP, b::Vector{SingleFilter}, a, rng::AbstractRNG)
    # s = rand(rng, b) # NOTE: Different than Josh's implementation
    s = FlatState([random_target_from_filter(rng, filter) for filter in b])
    # if isterminal(bmdp.pomdp, s)
    #   bp = bmdp_handle_terminal(bmdp.pomdp, bmdp.updater, b, s, a, rng::AbstractRNG)::typeof(b)
    #   return (sp=bp, r=0.0)
    # end
    sp, o = @gen(:sp, :o)(bmdp.pomdp, s, a, rng)
    bp = update(bmdp.updater, b, a, o)
    # r = bmdp.belief_reward(bmdp.pomdp, b, a, bp)
    r = POMDPs.reward(bmdp.pomdp, s, b, sp, bp)
    return (sp=bp, r=r)
end

# function rand(rng::AbstractRNG, b::Vector{SingleFilter})
#     if isempty(b)
#         return Target(1, 1.0, 42.0, 42.0, 42.0, 42.0)
#     end
#     return [random_target_from_filter(rng, filter) for filter in b]
# end

function BetaZero.accuracy(pomdp::FlatPOMDP, b0, s0, states, actions, returns)
    # Function to determine accuracy of agent's final decision.
    return POMDPs.reward(pomdp, s0, b0)
end

solver = BetaZeroSolver(;
    pomdp=pomdp,
    updater=up,
    params=BetaZeroParameters(; n_iterations=50, n_data_gen=50),
    nn_params=BetaZeroNetworkParameters(
        pomdp,
        up;
        training_epochs=50,
        n_samples=100_000,
        batchsize=1024,
        learning_rate=1e-4,
        λ_regularization=1e-5,
        use_dropout=true,
        p_dropout=0.2,
        use_prioritized_action_selection=false,
        optimizer=Flux.Adam, # Training optimizer (e.g., Adam, Descent, Nesterov)
        # use_epsilon_greedy=true, # Use epsilon-greedy exploration during action widening
        # ϵ_greedy=0.05,
    ),
    verbose=true,
    collect_metrics=true,
    plot_incremental_data_gen=true,
    mcts_solver=PUCTSolver(;
        n_iterations=100,
        k_action=2.0,
        alpha_action=0.25,
        k_state=2.0,
        alpha_state=0.1,
        enable_action_pw=true,
        enable_state_pw=true,
        counts_in_info=true, # Note, required for policy vector.
        final_criterion=MaxZQN(; zq=1, zn=1),
        exploration_constant=0.0,
    ),
)

policy = solve(solver, pomdp)
save_policy(policy, dir_paths.model_dir * "policy.bson")
save_solver(solver, dir_paths.model_dir * "solver.bson")

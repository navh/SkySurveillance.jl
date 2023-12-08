module SkySurveillance

using CUDA
using CommonRLInterface: AbstractEnv
using Dates: format, now
using Distributions: Normal, Uniform
using Flux
using Flux: params
using IntervalSets
using POMDPTools:
    Deterministic, HistoryRecorder, POMDPTools, RandomPolicy, RandomSolver, eachstep
using POMDPs: POMDP, POMDPs, Solver, Updater, discount, isterminal, reward, simulate, solve
using Plots: @animate, Plots, RGB, Shape, distinguishable_colors, mov, plot, plot!
using Random: AbstractRNG, Xoshiro
using ReinforcementLearning
using StaticArrays: SVector
using Statistics: mean, var
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

function RL.Experiment(
    ::Val{:JuliaRL},
    ::Val{:SAC},
    ::Val{:BeliefPOMDP},
    ::Nothing;
    save_dir=nothing,
    seed=PARAMS["seed"],
)
    rng = Xoshiro(seed)
    env = convert(AbstractEnv, pomdp)
    # action_dims = inner_env.n_actions
    action_dims = 1
    # A = action_space(inner_env)
    # low = A.left
    # high = A.right
    low = 0.0
    high = 0.0
    # ns = length(state(inner_env))
    ns = 2 * PARAMS["number_of_targets"]
    na = 1

    ## I don't think I need this as all my actions are always legal
    # env = ActionTransformedEnv(
    #     inner_env; action_mapping=x -> low + (x[1] + 1) * 0.5 * (high - low)
    # )
    init = glorot_uniform(rng)

    function create_policy_net()
        return gpu(
            NeuralNetworkApproximator(;
                model=GaussianNetwork(;
                    pre=Chain(
                        Dense(ns, 30, relu; init=init), Dense(30, 30, relu; init=init)
                    ),
                    μ=Chain(Dense(30, na; init=init)),
                    logσ=Chain(
                        Dense(
                            30, na, x -> clamp(x, typeof(x)(-10), typeof(x)(2)); init=init
                        ),
                    ),
                ),
                optimizer=Adam(0.003),
            ),
        )
    end

    function create_q_net()
        return gpu(
            NeuralNetworkApproximator(;
                model=Chain(
                    Dense(ns + na, 30, relu; init=init),
                    Dense(30, 30, relu; init=init),
                    Dense(30, 1; init=init),
                ),
                optimizer=Adam(0.003),
            ),
        )
    end

    agent = Agent(;
        policy=SACPolicy(;
            policy=create_policy_net(),
            qnetwork1=create_q_net(),
            qnetwork2=create_q_net(),
            target_qnetwork1=create_q_net(),
            target_qnetwork2=create_q_net(),
            γ=0.99f0,
            τ=0.005f0,
            α=0.2f0,
            batch_size=64,
            start_steps=1000,
            start_policy=RandomPolicy(pomdp; rng=rng),
            update_after=1000,
            update_freq=1,
            automatic_entropy_tuning=true,
            lr_alpha=0.003f0,
            action_dims=action_dims,
            rng=rng,
            device_rng=CUDA.functional() ? CUDA.CURAND.RNG() : rng,
        ),
        trajectory=CircularArraySARTTrajectory(;
            capacity=10000, state=Vector{Float32} => (ns,), action=Vector{Float32} => (na,)
        ),
    )

    stop_condition = StopAfterStep(10_000; is_show_progress=!haskey(ENV, "CI"))
    hook = TotalRewardPerEpisode()
    return Experiment(agent, env, stop_condition, hook, "# Radar with SAC")
end

ex = E`JuliaRL_SAC_BeliefPOMDP`
run(ex)
plot(ex.hook.rewards)

#### Animation begin 
#
# #solver = RandomMultiFilter()
# solver = RandomSolver()
# policy = solve(solver, pomdp)
#
# hr = HistoryRecorder(; max_steps=PARAMS["animation_steps"])
# history = simulate(hr, pomdp, policy)
#
# if PARAMS["render"]
#     anim = @animate for step in eachstep(history)
#         POMDPTools.render(pomdp, step)
#     end
#     mkpath(PARAMS["video_path"])
#     mov(anim, "$(PARAMS["video_path"])$(run_time).mov"; loop=1)
# end
#
# rewards = [reward(pomdp, step.s, step.b) for step in eachstep(history)]
#
# mkpath(PARAMS["log_path"])
# open("$(PARAMS["log_path"])$(run_time).txt", "w") do f
#     for i in rewards
#         println(f, i)
#     end
# end
#
#### Animation end 

#### Crux.jl experiment begin
#
# # Construct the Mujoco environment
# mdp = pomdp
# S = state_space(mdp)
# adim = length(POMDPs.actions(mdp)[1])
# amin = -1 * ones(Float32, adim)
# amax = 1 * ones(Float32, adim)
# rand_policy = FunctionPolicy((s) -> Float32.(rand.(Uniform.(amin, amax))))
#
# # Initializations that match the default PyTorch initializations
# Winit(out, in) = Float32.(rand(Uniform(-sqrt(1 / in), sqrt(1 / in)), out, in))
# binit(out, in) = Float32.(rand(Uniform(-sqrt(1 / in), sqrt(1 / in)), out))
#
# # Build the networks
# idim = S.dims[1] + adim
# # Networks for on-policy algorithms
# function μ()
#     return ContinuousNetwork(
#         Chain(
#             Dense(S.dims[1], 64, tanh; init=Winit, bias=binit(64, S.dims[1])),
#             Dense(64, 32, tanh; init=Winit, bias=binit(32, 64)),
#             Dense(32, adim; init=Winit, bias=binit(adim, 32)),
#         ),
#     )
# end
# function V()
#     return ContinuousNetwork(
#         Chain(
#             Dense(S.dims[1], 64, tanh; init=Winit, bias=binit(64, S.dims[1])),
#             Dense(64, 32; init=Winit, bias=binit(32, 64)),
#             Dense(32, 1; init=Winit, bias=binit(1, 32)),
#         ),
#     )
# end
# log_std() = -0.5f0 * ones(Float32, adim)
#
# # Networks for off-policy algorithms
# function Q()
#     return ContinuousNetwork(
#         Chain(
#             Dense(idim, 256, relu; init=Winit, bias=binit(256, idim)),
#             Dense(256, 256, relu; init=Winit, bias=binit(256, 256)),
#             Dense(256, 1; init=Winit, bias=binit(1, 256)),
#         ),
#     )
# end
# function A()
#     return ContinuousNetwork(
#         Chain(
#             Dense(S.dims[1], 256, relu; init=Winit, bias=binit(256, S.dims[1])),
#             Dense(256, 256, relu; init=Winit, bias=binit(256, 256)),
#             Dense(256, 6, tanh; init=Winit, bias=binit(6, 256)),
#         ),
#     )
# end
# function SAC_A()
#     base = Chain(
#         Dense(S.dims[1], 256, relu; init=Winit, bias=binit(256, S.dims[1])),
#         Dense(256, 256, relu; init=Winit, bias=binit(256, 256)),
#     )
#     mu = ContinuousNetwork(Chain(base..., Dense(256, 6; init=Winit, bias=binit(6, 256))))
#     logΣ = ContinuousNetwork(Chain(base..., Dense(256, 6; init=Winit, bias=binit(6, 256))))
#     return SquashedGaussianPolicy(mu, logΣ)
# end
#
# ## Setup params
# shared = (max_steps=1000, N=Int(3e6), S=S)
# on_policy = (
#     ΔN=4000,
#     λ_gae=0.97,
#     a_opt=(batch_size=4000, epochs=80, optimizer=Adam(3e-4)),
#     c_opt=(batch_size=4000, epochs=80, optimizer=Adam(1e-3)),
# )
# off_policy = (
#     ΔN=50,
#     max_steps=1000,
#     log=(period=4000, fns=[log_undiscounted_return(3)]),
#     buffer_size=Int(1e6),
#     buffer_init=1000,
#     c_opt=(batch_size=100, optimizer=Adam(1e-3)),
#     a_opt=(batch_size=100, optimizer=Adam(1e-3)),
#     π_explore=FirstExplorePolicy(
#         10000, rand_policy, GaussianNoiseExplorationPolicy(0.1f0; a_min=amin, a_max=amax)
#     ),
# )
#
# ## Run solvers 
# # 𝒮_ppo = PPO(;
# #     π=ActorCritic(GaussianPolicy(μ(), log_std()), V()), λe=0.0f0, shared..., on_policy...
# # )
# # solve(𝒮_ppo, mdp)
#
# # Solve with DDPG
# # 𝒮_ddpg = DDPG(; π=gpu(ActorCritic(A(), Q())), shared..., off_policy...)
# # solve(𝒮_ddpg, mdp)
#
# # Solve with TD3
# # 𝒮_td3 = TD3(;
# #     π=gpu(ActorCritic(A(), DoubleNetwork(Q(), Q()))),
# #     shared...,
# #     off_policy...,
# #     π_smooth=GaussianNoiseExplorationPolicy(
# #         0.2f0; ϵ_min=-0.5f0, ϵ_max=0.5f0, a_min=amin, a_max=amax
# #     ),
# # )
# # solve(𝒮_td3, mdp)
#
# # Solve with SAC
# 𝒮_sac = SAC(;
#     #π=gpu(ActorCritic(SAC_A(), DoubleNetwork(Q(), Q()))), shared..., off_policy...
#     π=cpu(ActorCritic(SAC_A(), DoubleNetwork(Q(), Q()))), shared..., off_policy...
# )
# solve(𝒮_sac, mdp)
# p = plot_learning([𝒮_sac]; title="Training Curves", labels=["SAC"])
# # Plot the learning curve
# # p = plot_learning(
# #     [𝒮_ppo, 𝒮_ddpg, 𝒮_td3, 𝒮_sac];
# #     title="HalfCheetah Mujoco Training Curves",
# #     labels=["PPO", "DDPG", "TD3", "SAC"],
# # )
# # Crux.savefig("examples/rl/half_cheetah_mujoco_benchmark.pdf")
# #### Crux.jl experiment end

end
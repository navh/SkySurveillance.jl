struct BicycleSolver <: Solver
    updater::Updater
end

function POMDPs.solve(::BicycleSolver, pomdp::POMDP)
    up = MultiFilterUpdater(pomdp.rng)
    return RandomPolicy(pomdp.rng, pomdp, up)
end

########################## experiment 

function experiment(rng, pomdp)
    inner_env = pomdp
    action_dims = inner_env.n_actions
    A = action_space(inner_env)
    low = A.left
    high = A.right
    ns = length(state(inner_env))
    na = 1

    env = ActionTransformedEnv(
        inner_env; action_mapping=x -> low + (x[1] + 1) * 0.5 * (high - low)
    )
    init = Flux.glorot_uniform(rng)

    function create_policy_net()
        return Approximator(
            SoftGaussianNetwork(
                Chain(Dense(ns, 30, relu; init=init), Dense(30, 30, relu; init=init)),
                Chain(Dense(30, na; init=init)),
                Chain(Dense(30, na, softplus; init=init)),
            ),
            Adam(0.003),
        )
    end

    function create_q_net()
        return TargetNetwork(
            Approximator(
                Chain(
                    Dense(ns + na, 30, relu; init=init),
                    Dense(30, 30, relu; init=init),
                    Dense(30, 1; init=init),
                ),
                Adam(0.003),
            );
            ρ=0.99f0,
        )
    end

    agent = Agent(;
        policy=SACPolicy(;
            policy=create_policy_net(),
            qnetwork1=create_q_net(),
            qnetwork2=create_q_net(),
            γ=0.99f0,
            α=0.2f0,
            start_steps=1000,
            start_policy=RandomPolicy(-1.0 .. 1.0; rng=rng),
            automatic_entropy_tuning=true,
            lr_alpha=0.003f0,
            action_dims=action_dims,
            rng=rng,
            device_rng=rng,
        ),
        trajectory=Trajectory(
            CircularArraySARTSTraces(;
                capacity=10000, state=Float32 => (ns,), action=Float32 => (na,)
            ),
            BatchSampler{SS′ART}(64),
            InsertSampleRatioController(; ratio=1 / 1, threshold=1000),
        ),
    )

    stop_condition = StopAfterStep(30_000; is_show_progress=!haskey(ENV, "CI"))
    hook = TotalRewardPerEpisode()
    return Experiment(agent, env, stop_condition, hook)
end

########################## sac
mutable struct SACPolicy{
    BA<:Approximator{<:SoftGaussianNetwork},
    BC1<:TargetNetwork,
    BC2<:TargetNetwork,
    P,
    R<:AbstractRNG,
    DR<:AbstractRNG,
} <: AbstractPolicy
    policy::BA
    qnetwork1::BC1
    qnetwork2::BC2
    γ::Float32
    α::Float32
    start_steps::Int
    start_policy::P
    automatic_entropy_tuning::Bool
    lr_alpha::Float32
    target_entropy::Float32
    update_step::Int
    rng::R
    device_rng::DR
    # Logging
    reward_term::Float32
    entropy_term::Float32
end

"""
    SACPolicy(;kwargs...)

# Keyword arguments

- `policy`, used to get action.
- `qnetwork1::TargetNetwork`, used to get Q-values.
- `qnetwork2::TargetNetwork`, used to get Q-values.
- `start_policy`, 
- `γ::Float32 = 0.99f0`, reward discount rate.
- `α::Float32 = 0.2f0`, entropy term.
- `start_steps = 10000`, number of steps where start_policy is used to sample actions
- `update_after = 1000`, number of steps before starting to update policy
- `automatic_entropy_tuning::Bool = true`, whether to automatically tune the entropy.
- `lr_alpha::Float32 = 0.003f0`, learning rate of tuning entropy.
- `action_dims = 0`, the dimensionality of the action. if `automatic_entropy_tuning = true`, must enter this parameter.
- `update_step = 0`,
- `rng = Random.default_rng()`, used to sample batch from trajectory or action from action distribution.
- `device_rng = Random.default_rng()`, should be set to `CUDA.CURAND.RNG()` if the `policy` is set to work with `CUDA.jl`

`policy` is expected to output a tuple `(μ, σ)` of mean and
standard deviations for the desired action distributions, this
can be implemented using a `SoftGaussianNetwork` in a `Approximator`.

Implemented based on http://arxiv.org/abs/1812.05905
"""
function SACPolicy(;
    policy,
    qnetwork1,
    qnetwork2,
    γ=0.99f0,
    α=0.2f0,
    start_steps=10000,
    automatic_entropy_tuning=true,
    lr_alpha=0.003f0,
    action_dims=0,
    update_step=0,
    start_policy=update_step == 0 ? identity : policy,
    rng=default_rng(),
    device_rng=default_rng(),
)
    if automatic_entropy_tuning
        @assert action_dims != 0
    end
    return SACPolicy(
        policy,
        qnetwork1,
        qnetwork2,
        γ,
        α,
        start_steps,
        start_policy,
        automatic_entropy_tuning,
        lr_alpha,
        Float32(-action_dims),
        update_step,
        rng,
        device_rng,
        0.0f0,
        0.0f0,
    )
end

function plan!(p::SACPolicy, pomdp)
    p.update_step += 1
    if p.update_step <= p.start_steps
        action = RLBase.plan!(p.start_policy, env)
    else
        D = device(p.policy)
        s = send_to_device(D, state(env))
        s = Flux.unsqueeze(s; dims=ndims(s) + 1)
        # trainmode:
        action
        send_to_host(action)

        # testmode:
        # if testing dont sample an action, but act deterministically by
        # taking the "increment" action
    end
end

function optimise!(p::SACPolicy, ::PostActStage, traj::Trajectory)
    for batch in traj
        update_critic!(p, batch)
        update_actor!(p, batch)
    end
end

function soft_q_learning_target(p::SACPolicy, r, t, s′)
    a′, log_π = RLCore.forward(
        p.policy, p.device_rng, s′; is_sampling=true, is_return_log_prob=true
    )
    q′_input = vcat(s′, a′)
    q′ = min.(target(p.qnetwork1)(q′_input), target(p.qnetwork2)(q′_input))

    return r .+ p.γ .* (1 .- t) .* dropdims(q′ .- p.α .* log_π; dims=1)
end

function q_learning_loss(qnetwork, s, a, y)
    q_input = vcat(s, a)
    q = dropdims(model(qnetwork)(q_input); dims=1)
    return mse(q, y)
end

function update_critic!(p::SACPolicy, batch::NamedTuple{SS′ART})
    s, s′, a, r, t = send_to_device(device(p.qnetwork1), batch)

    y = soft_q_learning_target(p, r, t, s′)

    # Train Q Networks
    q_grad_1 = gradient(Flux.params(model(p.qnetwork1))) do
        q_learning_loss(p.qnetwork1, s, a, y)
    end
    RLBase.optimise!(p.qnetwork1, q_grad_1)

    q_grad_2 = gradient(Flux.params(model(p.qnetwork2))) do
        q_learning_loss(p.qnetwork2, s, a, y)
    end
    return RLBase.optimise!(p.qnetwork2, q_grad_2)
end

function update_actor!(p::SACPolicy, batch::NamedTuple{SS′ART})
    s = send_to_device(device(p.qnetwork1), batch[:state])
    a = send_to_device(device(p.qnetwork1), batch[:action])

    # Train Policy
    p_grad = gradient(Flux.params(p.policy)) do
        a, log_π = RLCore.forward(
            p.policy, p.device_rng, s; is_sampling=true, is_return_log_prob=true
        )
        q_input = vcat(s, a)
        q = min.(model(p.qnetwork1)(q_input), model(p.qnetwork2)(q_input))
        reward = mean(q)
        entropy = mean(log_π)
        ignore_derivatives() do
            p.reward_term = reward
            p.entropy_term = entropy
            if p.automatic_entropy_tuning # Tune entropy automatically
                p.α -= p.lr_alpha * mean(-log_π .- p.target_entropy)
            end
        end
        p.α * entropy - reward
    end
    return RLBase.optimise!(p.policy, p_grad)
end
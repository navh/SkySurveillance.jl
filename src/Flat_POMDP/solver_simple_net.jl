# NOTE: Down on observation_summary_vector implementation, this used to keep zygote happy and now it is sad.
# I think it came in the midst of changing some f64 to f32, but it seems like it's a push!() problem wanting
# some sort of a zygote-specific buffer. 
# For now I am abandoning this entire approach in favor of a more 'conventional' MCTS with DPW from pomdps.jl 

@kwdef struct SimpleGreedySolver <: Solver
    pomdp::BeliefPOMDP
    n_epochs::Int
    n_train_episodes::Int
    n_test_episodes::Int
    n_skip_first_steps::Int
    n_steps_per_episode::Int
    monte_carlo_rollouts::Int
end

# function DenseRegularizedLayer(in_out::Pair, activation)
#     input, output = in_out
#     # if use_batchnorm && !use_dropout
#     # end
#     return [Dense(input => output, activation)]
#     # end
# end

function POMDPs.solve(solver::SimpleGreedySolver, pomdp::BeliefPOMDP)
    action_width = 1 # TODO: should be length(action) or based on types.jl somehow
    belief_width = 2 * pomdp.underlying_pomdp.number_of_targets
    input_width = action_width + belief_width

    # First, collect just a bunch of belief, action, scores
    model = Chain(
        Dense(input_width, 256, relu),
        Dense(256, 256),
        Dense(256, 256),
        Dense(256, 256),
        Dense(256, 1),
    )

    opt_state = Flux.setup(Adam(), model)

    loss_history = []
    for epoch_index in 1:(solver.n_epochs)
        for _ in 1:(solver.n_train_episodes)
            i = 0
            # child_pomdp = FlatPOMDP(rng, DISCOUNT)
            # u = MultiFilterUpdater(child_pomdp.rng)
            # pomdp = BeliefPOMDP(child_pomdp.rng, child_pomdp, u)

            for (a, o, r) in stepthrough(
                pomdp,
                RandomPolicy(pomdp.rng, pomdp, NothingUpdater()),
                "a,o,r";
                max_steps=solver.n_skip_first_steps + solver.n_steps_per_episode,
            )
                i += 1
                if i > solver.n_skip_first_steps
                    grads = Flux.gradient(model) do _
                        summary = observation_summary_vector(
                            o, pomdp.underlying_pomdp.number_of_targets
                        )
                        result = model(action_observation(a, summary))
                        # typeof(result)
                        mse(result, r)
                    end
                    Flux.update!(opt_state, model, grads[1])
                end
            end
        end
        sum_loss = 0.0
        count_loss = 0
        for _ in 1:(solver.n_test_episodes)
            i = 0
            # child_pomdp = FlatPOMDP(rng, DISCOUNT)
            # updater = MultiFilterUpdater(child_pomdp.rng)
            # pomdp = BeliefPOMDP(child_pomdp.rng, child_pomdp, updater)

            for (a, o, r) in stepthrough(
                pomdp,
                RandomPolicy(pomdp.rng, pomdp, NothingUpdater()),
                "a,o,r";
                max_steps=solver.n_skip_first_steps + solver.n_steps_per_episode,
            )
                i += 1
                if i > solver.n_skip_first_steps
                    summary = observation_summary_vector(
                        o, pomdp.underlying_pomdp.number_of_targets
                    )
                    result = model(action_observation(a, summary))
                    sum_loss += mse(result, r)
                    count_loss += 1
                end
            end
        end
        mean_loss = sum_loss / count_loss
        @info "epoch $(epoch_index) mean loss: $(mean_loss)"
        push!(loss_history, mean_loss)
    end

    #TODO: add saving logic 
    #TODO: add loading logic from here too
    #
    # mkpath(PARAMS["model_path"])
    # jldsave("$(run_time)-trained.jld2"; opt_state)

    return MCTwigSPolicy(
        model, solver.monte_carlo_rollouts, pomdp.underlying_pomdp.number_of_targets
    )
end

function filter_mean_θ(filter::SingleFilter)
    return mean([atan(particle.y, particle.x) for particle in filter.particles])
end

function observation_summary_vector(belief, number_of_targets)
    # The actual answer is https://arxiv.org/abs/1910.06764 , but padding will have to do for now

    θs_and_variances = sort([
        (filter_mean_θ(filter), filter_variance(filter)) for filter in belief
    ])
    # θs_and_variances = (
    #     (filter_mean_θ(filter), filter_variance(filter)) for filter in belief
    # )
    s = vcat(
        [θ for (θ, var) in θs_and_variances],
        zeros(Float32, number_of_targets - length(θs_and_variances)),
        [var for (θ, var) in θs_and_variances],
        zeros(Float32, number_of_targets - length(θs_and_variances)),
    )
    return s
end

function action_observation(action, observation)
    # return SVector{length(action) + length(observation),Float32}(vcat(action, observation))
    return Vector{Float32}(vcat(action, observation))
end

function monte_carlo_twig_search(model, observation, action_space, n_rollouts, max_targets)
    #todo - just jam this all in the action space because unpacking the entire policy this way is silly 

    summary = observation_summary_vector(observation, max_targets)
    best_action = rand(model.underlying_pomdp.rng, action_space)

    if sum(summary) == 0.0
        return best_action
    end

    best_estimate = model(action_observation(best_action, summary))
    for a in rand(model.underlying_pomdp.rng, action_space, n_rollouts)
        estimate = model(action_observation(a, summary))
        if estimate > best_estimate
            best_estimate = estimate
            best_action = a
        end
    end
    return best_action
end

struct MCTwigSPolicy{M<:Flux.Chain} <: Policy
    model::M
    n_rollouts::Int
    max_targets::Int
end

function POMDPs.updater(::MCTwigSPolicy)
    return PreviousObservationUpdater()
end

function POMDPs.action(p::MCTwigSPolicy, b::SVector)
    # TODO: replace uniform(0,1) with some action_space(p.model) or something
    return monte_carlo_twig_search(p.model, b, Uniform(0, 1), p.n_rollouts, p.max_targets)
end

function POMDPs.action(p::MCTwigSPolicy, b)
    return 0.0
end

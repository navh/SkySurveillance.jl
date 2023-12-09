struct SimpleGreedySolver <: Solver end

N_EPOCHS = 1
N_TRAIN_EPISODES = 100
N_TEST_EPISODES = 10
N_SKIP_FIRST_STEPS = 100
N_STEPS_PER_EPISODE = 100
MONTE_CARLO_ROLLOUTS = 10

ACTION_WIDTH = 1
BELIEF_WIDTH = 2 * PARAMS["number_of_targets"]
INPUT_WIDTH = ACTION_WIDTH + BELIEF_WIDTH

function POMDPs.solve(::SimpleGreedySolver, pomdp::POMDP)

    # First, collect just a bunch of belief, action, scores
    model = Chain(Dense(INPUT_WIDTH, 256, relu), Dense(256, 256), Dense(256, 1))

    opt_state = Flux.setup(Adam(), model)

    loss_history = []
    for epoch_index in 1:N_EPOCHS
        for _ in 1:N_TRAIN_EPISODES
            i = 0
            child_pomdp = FlatPOMDP(rng, DISCOUNT)
            u = MultiFilterUpdater(child_pomdp.rng)
            pomdp = BeliefPOMDP(child_pomdp.rng, child_pomdp, u)

            for (a, o, r) in stepthrough(
                pomdp,
                RandomPolicy(pomdp.rng, pomdp, NothingUpdater()),
                "a,o,r";
                max_steps=N_SKIP_FIRST_STEPS + N_STEPS_PER_EPISODE,
            )
                i += 1
                if i > N_SKIP_FIRST_STEPS
                    grads = Flux.gradient(model) do m
                        result = model(action_observation(a, o))
                        mse(result, r)
                    end
                    Flux.update!(opt_state, model, grads[1])
                end
            end
        end
        sum_loss = 0.0
        count_loss = 0
        for _ in 1:N_TEST_EPISODES
            i = 0
            child_pomdp = FlatPOMDP(rng, DISCOUNT)
            updater = MultiFilterUpdater(child_pomdp.rng)
            pomdp = BeliefPOMDP(child_pomdp.rng, child_pomdp, updater)

            for (a, o, r) in stepthrough(
                pomdp,
                RandomPolicy(pomdp.rng, pomdp, NothingUpdater()),
                "a,o,r";
                max_steps=N_SKIP_FIRST_STEPS + N_STEPS_PER_EPISODE,
            )
                i += 1
                if i > N_SKIP_FIRST_STEPS
                    result = model(action_observation(a, o))
                    sum_loss += mse(result, r)
                    count_loss += 1
                end
            end
        end
        mean_loss = sum_loss / count_loss
        @info "epoch $(epoch_index) mean loss: $(mean_loss)"
        push!(loss_history, mean_loss)
    end

    return MCTwigSPolicy(model)
end

function action_observation(action, observation)
    # return SVector{INPUT_WIDTH,Float32}(vcat(action, observation))
    return Vector{Float32}(vcat(action, observation))
end

function monte_carlo_twig_search(model, observation, action_space)
    best_action = rand(action_space)
    best_estimate = model(action_observation(best_action, observation))
    for a in rand(action_space, MONTE_CARLO_ROLLOUTS)
        estimate = model(action_observation(a, observation))
        if estimate > best_estimate
            best_estimate = estimate
            best_action = a
        end
    end
    return best_action
end

struct MCTwigSPolicy{M<:Flux.Chain} <: Policy
    model::M
end

function POMDPs.updater(::MCTwigSPolicy)
    return PreviousObservationUpdater()
end

function POMDPs.action(p::MCTwigSPolicy, b::SVector)
    # TODO: replace uniform(0,1) with some action_space(p.model) or something
    return monte_carlo_twig_search(p.model, b, Uniform(0, 1))
end

function POMDPs.action(p::MCTwigSPolicy, b)
    return 0.0
end

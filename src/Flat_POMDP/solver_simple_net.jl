struct SimpleGreedySolver <: Solver end

N_TRAIN_EPISODES = 50000
N_STEPS_PER_EPISODE = 100
SKIP_FIRST_STEPS = 100

ACTION_WIDTH = 1
BELIEF_WIDTH = 2 * PARAMS["number_of_targets"]
INPUT_WIDTH = ACTION_WIDTH + BELIEF_WIDTH

function POMDPs.solve(::SimpleGreedySolver, pomdp::POMDP)

    # First, collect just a bunch of belief, action, scores
    model = Chain(Dense(INPUT_WIDTH, 256, relu), Dense(256, 256), Dense(256, 1))

    opt_state = Flux.setup(Adam(), model)

    for _ in N_TRAIN_EPISODES
        i = 0
        for (a, o, r) in stepthrough(
            pomdp,
            RandomPolicy(pomdp.rng, pomdp, NothingUpdater()),
            "a,o,r";
            max_steps=N_STEPS_PER_EPISODE,
        )
            i += 1
            if i > SKIP_FIRST_STEPS
                belief_action = SVector{INPUT_WIDTH,Float32}(vcat(a, o))
                grads = Flux.gradient(model) do m
                    result = m(belief_action)
                    mse(result, r)
                end
                Flux.update!(opt_state, model, grads[1])
            end
        end
    end

    # Next, train a model to predict score given belief and action

    # This trained model is then used to create a policy that just 
    # evaluates the current belief with a few possible actions, 
    # and selects the best

    return 42
end

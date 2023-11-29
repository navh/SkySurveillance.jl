# Note: fig from "fully illuminated guess", nothing to do with FIB, I'll probably rename due to similarity

struct FigOfflineSolver <: Solver end

function POMDPs.solve(::FigOfflineSolver, pomdp::POMDP)

    #TODO: train RNN+decoder here?
    figup = train(pomdp)
    figup_state = Flux.state(figup)
    jldsave("./figup5.jld2"; figup_state)
    #figup = load_model("figup4.jld2")
    #
    up = FigUpdater(figup)
    return RandomPolicy(pomdp.rng, pomdp, up)
end

struct FigUpdater <: POMDPs.Updater
    model
end

function POMDPs.initialize_belief(up::FigUpdater, d)
    Flux.reset!(up.model)
    return zeros(Cells)
end

function POMDPs.update(up::FigUpdater, b, a, o)
    if isempty(o)
        return Cells(up.model(SVector{5,Float32}(a, 0.0, 0.0, 0.0, 0.0)))
    end
    guesses = [
        up.model(
            SVector{5,Float32}(
                a,
                1.0,
                target.r / PARAMS["radar_max_range_meters"],
                target.θ / π,
                target.v / √(2 * TARGET_VELOCITY_MAX_METERS_PER_SECOND^2),
            ), # The second 1.0 represents that a measurement has been made
        ) for target in o
    ]
    return Cells(last(guesses))
end

### nets.jl copy paste below

# based on https://github.com/FluxML/model-zoo/blob/master/text/char-rnn/char-rnn.jl
function build_model(input_width::Int, output_width::Int)
    h = 128
    return Chain(LSTM(input_width => h), LSTM(h => h), Dense(h => output_width))
end

function load_model(filename)
    device = PARAMS["use_gpu"] ? gpu : cpu
    model = device(build_model(5, length(Cells)))
    model_state = load(filename)
    Flux.loadmodel!(model, model_state)
    return model_state
end

# TODO: can this be done on the fly? Flux's docs don't make it obvious... this feels foolish
function get_data(pomdp::FlatPOMDP)
    @info "Generating $(PARAMS["train_sequences"]) train sequences and $(PARAMS["test_sequences"]) test sequences."
    Xs = []
    Ys = []

    for sequence_number in 1:(PARAMS["train_sequences"] + PARAMS["test_sequences"])
        # TODO: check rng, 'split' the RNG like JAX? does it do this automagically?
        rng = Xoshiro(PARAMS["seed"] + sequence_number)

        pomdp = FlatPOMDP(; rng=rng)

        solver = RandomSolver(rng)
        policy = solve(solver, pomdp)

        hr = HistoryRecorder(; max_steps=PARAMS["steps_per_sequence"])
        history = simulate(hr, pomdp, policy)
        X = []
        Y = []
        for (s, a, o) in eachstep(history, "(s,a,o)")
            occupancy_matrix = real_occupancy(s)
            if isempty(o)
                push!(Y, occupancy_matrix)
                push!(X, SVector{5,Float32}(a, 0.0, 0.0, 0.0, 0.0))
            else
                for target in o
                    push!(Y, occupancy_matrix)
                    push!(
                        X,
                        SVector{5,Float32}(
                            a,
                            1.0,
                            target.r / PARAMS["radar_max_range_meters"],
                            target.θ / π,
                            target.v / √(2 * TARGET_VELOCITY_MAX_METERS_PER_SECOND^2),
                        ), # The second 1.0 represents that a measurement has been made
                    )
                end
            end
        end
        push!(Xs, X)
        push!(Ys, Y)
    end
    return (
        Xs[1:PARAMS["train_sequences"]],
        Ys[1:PARAMS["train_sequences"]],
        Xs[(PARAMS["train_sequences"] + 1):end],
        Ys[(PARAMS["train_sequences"] + 1):end],
    )
end

function train(pomdp::FlatPOMDP)
    device = PARAMS["use_gpu"] ? gpu : cpu
    @info "Beginning training on $(device)"

    ## Get Data
    @info "Get Data"
    (trainX, trainY, testX, testY) = get_data(pomdp)

    @info "Moving data to $(device)"
    trainX, trainY, testX, testY = device.((trainX, trainY, testX, testY))

    @info "Constructing Model"
    model = device(build_model(length(trainX[1][1]), length(Cells)))

    # function loss(m, xs, ys)
    #     Flux.reset!(m)
    #     return sum(logitcrossentropy.([m(x) for x in xs], ys))
    # end

    function loss(m, xs, ys)
        Flux.reset!(m)
        return sum(mse.([m(x) for x in xs], ys, agg=sum))
    end

    ## Training
    opt_state = Flux.setup(Adam(PARAMS["learning_rate"]), model)

    for epoch in 1:PARAMS["epochs"]
        @info "Training, epoch $(epoch) / $(PARAMS["epochs"])"
        Flux.train!(loss, model, zip(trainX, trainY), opt_state)
        if PARAMS["write_model"]
            jldsave("figup-nov28-e$(epoch).jld2"; model_state=Flux.state(model))
        end

        ## Show loss-per-step over the test set
        @info "loss-per-step $(sum(loss.(Ref(model), testX, testY)) / (PARAMS["steps_per_sequence"] * length(testX)))"
    end
    return model
end

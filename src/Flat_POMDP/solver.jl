# Note: fig from "fully illuminated guess", nothing to do with FIB, I'll probably rename due to similarity

struct FigOfflineSolver <: Solver end

function POMDPs.solve(::FigOfflineSolver, pomdp::POMDP)

    #TODO: train RNN+decoder here?
    figup = train(pomdp)
    figup_state = Flux.state(figup)
    jldsave("$(PARAMS["model_path"])-figup.jld2"; figup_state)
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
    h = 256
    return Chain(
        LSTM(input_width => h), LSTM(h => h), Dense(h => h), Dense(h => output_width)
    )
end

# TODO: can this be done on the fly? Flux's docs don't make it obvious... this feels foolish
# Answer: Yes: data must support the 'getobs' method https://fluxml.ai/Flux.jl/stable/data/mlutils/#MLUtils.DataLoader
function get_data(pomdp::FlatPOMDP, sequence_count::Int64, device)
    @info "Generating $(sequence_count) sequences"

    data = []
    # Ugh, ugly svector declarations to make it play nice with CUDA
    x_type = SVector{5,Float32}
    y_type = SVector{length(Cells),Float32} # Apparently large StaticArrays stresses the compiler and does more harm than good

    #for sequence_number in 1:(PARAMS["train_sequences"] + PARAMS["test_sequences"])
    for sequence_number in 1:sequence_count
        # TODO: check rng, 'split' the RNG like JAX? does it do this automagically?
        rng = Xoshiro(PARAMS["seed"] + sequence_number)
        x = []
        y = []

        pomdp = FlatPOMDP(; rng=rng)

        solver = RandomSolver(rng)
        policy = solve(solver, pomdp)
        hr = HistoryRecorder(; max_steps=PARAMS["steps_per_sequence"])
        history = simulate(hr, pomdp, policy)
        for (s, a, o) in eachstep(history, "(s,a,o)")
            occupancy_matrix = real_occupancy(s)
            if isempty(o)
                push!(x, x_type(a, 0.0, 0.0, 0.0, 0.0))
                push!(y, y_type(occupancy_matrix))
            else
                for target in o
                    push!(
                        x,
                        x_type(
                            a,
                            1.0,
                            target.r / PARAMS["radar_max_range_meters"],
                            target.θ / π,
                            target.v / √(2 * TARGET_VELOCITY_MAX_METERS_PER_SECOND^2),
                        ), # second 1.0 represents that a measurement has been made
                    )
                    push!(y, y_type(occupancy_matrix))
                end
            end
        end
        x = x[1:PARAMS["steps_per_sequence"]]
        y = y[1:PARAMS["steps_per_sequence"]]
        x = device(x)
        y = device(y)
        push!(data, (x, y))
    end
    #data = xy_tuple_type(data)
    return device(data)
end

function train(pomdp::FlatPOMDP)
    device = if PARAMS["use_gpu"]
        gpu
    else
        cpu
    end
    @info "Beginning training on $(device)"

    ## Get Data
    @info "Get Data"
    trainData = get_data(pomdp, PARAMS["train_sequences"], device)
    testData = get_data(pomdp, PARAMS["test_sequences"], device)

    @info "Constructing Model"
    model = device(build_model(5, length(Cells)))

    # function loss(m, xs, ys)
    #     Flux.reset!(m)
    #     return sum(logitcrossentropy.([m(x) for x in xs], ys))
    # end
    ZEST = PARAMS["xy_bins"]^2 / PARAMS["number_of_targets"]

    function loss(m, xs, ys)
        Flux.reset!(m)
        return sum(mse.([m(x) * ZEST for x in xs], ys * ZEST)) / length(xs)
    end

    ## Training
    opt_state = Flux.setup(Adam(PARAMS["learning_rate"]), model)

    train_loss_history = Float32[]
    test_loss_history = Float32[]

    for epoch in 1:PARAMS["epochs"]
        @info "Training, epoch $(epoch) / $(PARAMS["epochs"])"
        Flux.train!(loss, model, trainData, opt_state)
        train_loss =
            sum(loss.(Ref(model), [x[1] for x in trainData], [x[2] for x in trainData])) / length(trainData)
        push!(train_loss_history, train_loss)
        @info "train_loss: $(train_loss)"

        test_loss =
            sum(loss.(Ref(model), [x[1] for x in testData], [x[2] for x in testData])) / length(testData)
        push!(test_loss_history, test_loss)
        @info "test_loss: $(test_loss)"

        if PARAMS["write_model"]
            if PARAMS["use_gpu"]
                cpu(model) # Dump it onto the CPU to save 
            end
            jldsave("$(PARAMS["model_path"])-$(epoch).jld2"; model_state=Flux.state(model))
            if PARAMS["use_gpu"]
                gpu(model)
            end
        end

        ## Show loss-per-step over the test set
        #@info "loss-per-step $(sum(loss.(Ref(model),[x[1] for x in testData],[x[2] for x in testData])) / PARAMS["test_sequences"]) "
    end
    @info "train_loss_history"
    @info train_loss_history
    @info "test_loss_history"
    @info test_loss_history
    return model
end
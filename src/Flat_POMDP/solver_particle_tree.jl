# Particle Filter Tree - Double Progressive Widening
# based on https://github.com/WhiffleFish/ParticleFilterTrees.jl/tree/main
# and https://github.com/sisl/BetaZero.jl/blob/main/src/BetaZero.jl 

Base.@kwdef struct ParticleTreeSolver{RNG<:AbstractRNG} <: POMDPs.Solver
    pomdp::POMDP
    updater::POMDPs.updater
    n_iterations::Int = 10
    tree_queries::Int = 1_000
    max_time::Float64 = Inf # (seconds)
    max_depth::Int = 20
    k_o::Float64 = 10.0
    α_o::Float64 = 0.0 # Observation Progressive widening parameter
    k_a::Float64 = 5.0
    α_a::Float64 = 0.0 # Action Progressive widening parameter
end

function POMDPs.solve(solver::ParticleTreeSolver, pomdp::FlatPOMDP)
    act = actions(pomdp)

    # solved_ve = MCTS.convert_estimator(solver.value_estimator, solver, pomdp)
    solved_value_estimator = 42

    # cache = BeliefCache{S}(sol)
    return PFTDPWPlanner(
        pomdp, solver, ParticleTree{S,A,O}(solver.tree_queries, sol.k_o, sol.k_a), solved_ve
    )
end

### TREE 

mutable struct PUCTTree{BeliefType,ActionType}
    # for each belief node
    total_n::Vector{Int}
    children::Vector{Vector{Int}}
    b_labels::Vector{BeliefType}

    # for each belief-action node
    a_labels::Vector{ActionType}
    q::Vector{Float64}

    function PUCTTree{StateType,ActionType}(size::Int=1000) where {StateType,ActionType}
        size = min(size, 100_000)
        return new(
            sizehint!(Int[], size),
            sizehint!(Vector{Int}[], size),
            sizehint!(StateType[], size),
            sizehint!(ActionType[], size),
            sizehint!(Float64[], size),
        )
    end
end

### PLANNER

struct ParticleTreePlanner{POMDP<:FlatPOMDP,SOLVER<:ParticleTreeSolver,VE} <: Policy
    pomdp::POMDP
    solver::SOLVER
    value_estimator::VE
end

# use value_estimator and mcts to determine best action
function POMDPs.action(planner::ParticleTreePlanner, b)
    A = actiontype(planner.pomdp)
    tree = PUCTTree{MultiFilterBelief,A}(planner.solver.n_iterations)
    bnode = insert_belief_node!(tree, b)

    # timer = p.solver.timer
    # p.solver.show_progress ? progress = Progress(p.solver.n_iterations) : nothing
    nquery = 0
    # start_s = timer()
    for i in 1:(planner.solver.n_iterations)
        nquery += 1
        simulate(planner, tree, planner.solver.max_depth) # (not 100% sure we need to make a copy of the state here)
        # p.solver.show_progress ? next!(progress) : nothing
        # if timer() - start_s >= p.solver.max_time
        #     p.solver.show_progress ? finish!(progress) : nothing
        #     break
        # end
    end

    banode = select_best(tree)
    a = tree.a_labels[banode] # choose action with highest approximate value
    return a
end

function simulate(planner, tree, depth)
    if depth = 0
    end
    # for i in 1:depth
    #     widen_actions!()
    #     widen_beliefs!()
    # end
end

function select_best(tree::PUCTTree)
    # I'm never revisiting things, so I can't take advantage of 'count'
    best_Q = -Inf
    sanode = 0
    for child in tree.children[0]
        if tree.q[child] > best_Q
            best_Q = tree.q[child]
            sanode = child
        end
    end
    return sanode
end

function insert_belief_node!(tree::PUCTTree, b::MultiFilterBelief)
    push!(tree.total_n, 0)
    push!(tree.children, Int[])
    push!(tree.b_labels, b)
    bnode = length(tree.total_n)
    return bnode
end

function insert_action_node!(tree::PUCTTree, bnode::Int, a::A, q0::Float64)
    push!(tree.a_labels, a)
    push!(tree.q, q0)
    banode = length(tree.a_labels)
    push!(tree.children[bnode], banode)
    return banode
end

## Neural Networks

function input_representation(belief)
    #TODO: turn the belief into something of constant size
    return observation_summary_vector(belief, 20)
end

"""
Get belief representation for network input, add batch dimension for Flux.
"""
function network_input(belief)
    b = Float32.(input_representation(belief))
    return Flux.unsqueeze(b; dims=ndims(b) + 1) # add extra single dimension (batch)
end

function initialize_network(input_size, layer_size, activation, action_size)
    function DenseRegularizedLayer(in_out::Pair)
        input, output = in_out
        # if use_batchnorm && !use_dropout
        # end
        return [Dense(input => output, activation)]
        # end
    end

    return Chain(
        DenseRegularizedLayer(prod(input_size) => layer_size)...,
        DenseRegularizedLayer(layer_size => layer_size)...,
        DenseRegularizedLayer(layer_size => layer_size)...,
        Parallel(
            vcat;
            value_head=Chain(
                DenseRegularizedLayer(layer_size => layer_size)...,
                Dense(layer_size => 1),
                # Note: A normalization layer will be added during training (with the old layer removed before the next training phase).
            ),
            policy_head=Chain(
                DenseRegularizedLayer(layer_size => layer_size)...,
                Dense(layer_size => action_size),
                softmax,
            ),
        ),
    )
end

softmax(x::AbstractArray{T}; dims=1) where {T} = softmax!(similar(x, float(T)), x; dims)

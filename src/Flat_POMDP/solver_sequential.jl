@kwdef struct SequentialSolver <: Solver end

function POMDPs.solve(solver::SequentialSolver, pomdp::BeliefPOMDP)
    min_beam_θ = -π
    max_beam_θ = π
    beams_to_cover_area = (max_beam_θ - min_beam_θ) / pomdp.underlying_pomdp.beamwidth_rad
    return IncrementerPolicy(beams_to_cover_area)
end

function action_observation(action, observation)
    return Vector{Float32}(vcat(action, observation))
end

struct IncrementerPolicy <: Policy
    increment_step::Number
end

function POMDPs.updater(::MCTwigSPolicy)
    return LastActionUpdater()
end

function POMDPs.action(p::MCTwigSPolicy, b)
    return (b + p.increment_step) % 1.0
end

struct LastActionUpdater <: POMDPs.Updater end

function POMDPs.initialize_belief(up::LastActionUpdater, a)
    return a
end

function POMDPs.update(up::LastActionUpdater, b, a, o)
    return a
end
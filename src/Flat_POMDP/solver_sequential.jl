struct SequentialSolver <: Solver end

function POMDPs.solve(solver::SequentialSolver, pomdp::BeliefPOMDP)
    min_beam_θ = -π
    max_beam_θ = π
    increment_step = pomdp.underlying_pomdp.beamwidth_rad / (max_beam_θ - min_beam_θ)
    return IncrementerPolicy(increment_step)
end

struct IncrementerPolicy <: Policy
    increment_step::Number
end

function POMDPs.updater(::IncrementerPolicy)
    return LastActionUpdater()
end

function POMDPs.action(p::IncrementerPolicy, b)
    new_action = (b + p.increment_step) % 1.0
    return new_action
end

struct LastActionUpdater <: POMDPs.Updater end

function POMDPs.initialize_belief(up::LastActionUpdater, state)
    return 0.0
end

function POMDPs.update(up::LastActionUpdater, b, a, o)
    return a
end

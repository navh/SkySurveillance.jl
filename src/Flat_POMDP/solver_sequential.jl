struct SequentialSolver <: Solver end

function POMDPs.solve(_::SequentialSolver, pomdp::BeliefPOMDP)
    return IncrementerPolicy(pomdp.underlying_pomdp.beamwidth_rad / 2π)
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

function POMDPs.initialize_belief(_::LastActionUpdater, _)
    return 0.0
end

function POMDPs.update(_::LastActionUpdater, _, a, _)
    return a
end

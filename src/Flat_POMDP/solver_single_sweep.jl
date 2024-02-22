struct SingleSweepSolver <: Solver end

function POMDPs.solve(_::SingleSweepSolver, pomdp::BeliefPOMDP)
    return SingleSweepPolicy(pomdp.underlying_pomdp.beamwidth_rad / 2π)
end

struct SingleSweepPolicy <: Policy
    increment_step::Number
end

function POMDPs.updater(::SingleSweepPolicy)
    return LastActionUpdater()
end

function POMDPs.action(p::SingleSweepPolicy, b)
    new_action = (b + p.increment_step)
    return min(1.0, new_action)
end

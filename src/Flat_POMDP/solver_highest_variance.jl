struct HighestVarianceSolver <: Solver end

function POMDPs.solve(solver::HighestVarianceSolver, pomdp::BeliefPOMDP)
    min_beam_θ = -π
    max_beam_θ = π
    increment_step = pomdp.underlying_pomdp.beamwidth_rad / (max_beam_θ - min_beam_θ)
    return HighestVariancePolicy(increment_step)
end

struct HighestVariancePolicy <: Policy
    random_beam_ratio::Number
end

function POMDPs.updater(::HighestVariancePolicy)
    return NothingUpdater()
end

function POMDPs.action(p::HighestVariancePolicy, b)
    @info b
    # new_action = (b + p.increment_step) % 1.0
    return 0
end

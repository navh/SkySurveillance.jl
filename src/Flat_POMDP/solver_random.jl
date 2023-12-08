struct RandomMultiFilter <: Solver end

function POMDPs.solve(::RandomMultiFilter, pomdp::POMDP)
    up = MultiFilterUpdater(pomdp.rng)
    return RandomPolicy(pomdp.rng, pomdp, up)
end
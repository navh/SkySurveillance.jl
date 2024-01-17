struct RandomMultiFilter <: Solver end

function POMDPs.solve(::RandomMultiFilter, pomdp::POMDP)
    up = MultiFilterUpdater(pomdp.rng, pomdp.underlying_pomdp.n_particles)
    return RandomPolicy(pomdp.rng, pomdp, up)
end

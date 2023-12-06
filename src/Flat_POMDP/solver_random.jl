# Note: fig from "fully illuminated guess", nothing to do with FIB, I'll probably rename due to similarity

struct RandomMultiFilter <: Solver end

function POMDPs.solve(::RandomMultiFilter, pomdp::POMDP)
    up = MultiFilterUpdater(pomdp.rng)
    return RandomPolicy(pomdp.rng, pomdp, up)
end
struct HighestVarianceSolver <: Solver end

function POMDPs.solve(solver::HighestVarianceSolver, pomdp::BeliefPOMDP)
    return HighestVariancePolicy(pomdp.rng, POMDPs.actions(pomdp), 0.5)
end

struct HighestVariancePolicy <: Policy
    rng::AbstractRNG
    action_space::Sampleable
    random_action_ratio::Number
end

function POMDPs.updater(::HighestVariancePolicy)
    return PreviousObservationUpdater()
end

function POMDPs.action(p::HighestVariancePolicy, initial::Deterministic{UpdaterState})
    # Deterministic{Main.SkySurveillance.UpdaterState}(
    #     Main.SkySurveillance.UpdaterState(
    #         Main.SkySurveillance.FlatState(
    #             Main.SkySurveillance.Target[
    #                 Main.SkySurveillance.Target(
    #                     1, 0.0f0, -35809.266f0, -385005.06f0, -425.6365f0, 625.91895f0
    #                 ),
    #                 Main.SkySurveillance.Target(
    #                     2, -48.99765f0, 38113.96f0, 288146.28f0, 256.33234f0, -409.58362f0
    #                 ),
    #             ],
    #         ),
    #         0.0,
    #         Main.SkySurveillance.TargetObservation[],
    #         Main.SkySurveillance.SingleFilter[],
    #     ),
    # )
    return 0.0
end

function POMDPs.action(p::HighestVariancePolicy, b)
    if isempty(b) || rand(p.rng) > p.random_action_ratio
        return rand(p.rng, p.action_space)
    end

    highest_variance = -Inf
    θ = Nothing
    for filter in b
        if filter_variance(filter) > highest_variance
            highest_variance = filter_variance(filter)

            # Yeah so even with very large (90degree) beams, it still loses the plot here. 
            # I think I should dial up the variance on the filters.
            # Might be smarter to sample some random particle instead of doing mean theta
            # This would hopefully allow it to wander around and explore the edges of very big clouds.
            θ = filter_mean_θ(filter)
        end
    end
    return (θ + π) / 2π # TODO: make some sort of "theta to action" thing, or just make action space -pi to pi, I mess this up every time 
end

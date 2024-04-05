# Inspired by Josh Ott: https://github.com/sisl/SBO_AIPPMS/blob/main/Rover/GP_BMDP_Rover/belief_mdp.jl
"""
    BeliefPOMDP(pomdp, updater)
Create a POMDP corresponding to POMDP `underlying_pomdp` with observations resulting from `updater`
"""

struct BeliefPOMDP <: POMDP{UpdaterState,FlatAction,UpdaterObservation}
    rng::AbstractRNG
    underlying_pomdp::FlatPOMDP
    updater::Updater
end

function POMDPs.isterminal(pomdp::BeliefPOMDP, s::UpdaterState)
    return isterminal(pomdp.underlying_pomdp, s.underlying_state)
end

function POMDPs.initialobs(pomdp::BeliefPOMDP, s::UpdaterState)
    return Deterministic(zeros(SVector{2 * pomdp.underlying_pomdp.number_of_targets}))
end

function POMDPs.gen(
    pomdp::BeliefPOMDP, s::UpdaterState, a::FlatAction, rng::RNG
) where {RNG<:AbstractRNG}
    sp = generate_s(pomdp, s, a, rng)
    o = generate_o(pomdp, s, a, sp, rng)
    r = reward(pomdp, s, a, sp)
    return (sp=sp, o=o, r=r)
end

### discount 

function POMDPs.discount(pomdp::BeliefPOMDP)
    return pomdp.underlying_pomdp.discount
end

### states 

function POMDPs.initialstate(pomdp::BeliefPOMDP)
    return Deterministic(
        UpdaterState(
            rand(pomdp.rng, POMDPs.initialstate(pomdp.underlying_pomdp)),
            0.0,
            TargetObservation[],
            POMDPs.initialize_belief(pomdp.updater),
        ),
    )
end

### actions

function POMDPs.actions(pomdp::BeliefPOMDP)
    return POMDPs.actions(pomdp.underlying_pomdp)
end

### observations

function generate_o(
    pomdp::BeliefPOMDP,
    s::UpdaterState,
    action::FlatAction,
    sp::UpdaterState,
    rng::AbstractRNG,
)
    return s.belief_state
    # return belief_to_observation(pomdp, s.belief_state)
end

### transitions

function generate_s(
    pomdp::BeliefPOMDP, s::UpdaterState, a::FlatAction, rng::RNG
) where {RNG<:AbstractRNG}
    sp = generate_s(pomdp.underlying_pomdp, s.underlying_state, a, rng)
    o = generate_o(pomdp.underlying_pomdp, s.underlying_state, a, sp, rng)
    bp = POMDPs.update(pomdp.updater, s.belief_state, a, o)
    return UpdaterState(sp, a, o, bp)
end

### rewards

# function score_tracked_target(target, filter)
#     # Basically RMS 
#     return sum([
#         √((target.x - particle.x)^2 + (target.y - particle.y)^2) for
#         particle in filter.particles
#     ]) / length(filter.particles)
# end
#
# function score_untracked_target(target)
#     return max(0, target.appears_at_t * 1e3)
# end

function POMDPs.reward(pomdp::BeliefPOMDP, s::UpdaterState, a)
    # bp = POMDPs.update(pomdp.updater, s.belief_state, a)
    # score = 0
    #
    # for target in s.underlying_state.targets
    #     if target.appears_at_t >= 0 &&
    #         √(target.x^2 + target.y^2) <= pomdp.underlying_pomdp.radar_max_range_meters
    #         filter_index = findfirst(x -> x.id == target.id, s.belief_state)
    #         # change to spread of the particle filter 
    #         if filter_index === nothing
    #             score += score_untracked_target(target)
    #         else
    #             score += score_tracked_target(target, s.belief_state[filter_index])
    #         end
    #     end
    # end
    #
    return POMDPs.reward(pomdp.underlying_pomdp, s.underlying_state, s.belief_state)
end

### visualizations 

function POMDPTools.render(pomdp::BeliefPOMDP, step::NamedTuple)
    underlying_step = (
        s=step.s.underlying_state,
        a=step.s.past_action,
        o=step.s.underlying_observation,
        b=step.s.belief_state,
    )
    return POMDPTools.render(pomdp.underlying_pomdp, underlying_step)
end

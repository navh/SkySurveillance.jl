# Inspired by Josh Ott: https://github.com/sisl/SBO_AIPPMS/blob/main/Rover/GP_BMDP_Rover/belief_mdp.jl
"""
    BeliefPOMDP(pomdp, updater)
Create a POMDP corresponding to POMDP `underlying_pomdp` with observations resulting from `updater`
"""

struct BeliefPOMDP <: POMDP{UpdaterState,FlatAction,UpdaterObservation}
    rng::AbstractRNG
    underlying_pomdp::POMDP
    updater::Updater
end

function POMDPs.isterminal(pomdp::BeliefPOMDP, s::UpdaterState)
    return isterminal(pomdp.underlying_pomdp, s.underlying_state)
end

function POMDPs.initialobs(pomdp::BeliefPOMDP, s::UpdaterState)
    return Deterministic(zeros(SVector{2 * PARAMS["number_of_targets"]}))
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
    return Deterministic(UpdaterState(initialize_random_targets(pomdp.rng), SingleFilter[]))
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
    # return s.belief_state
    return belief_to_observation(s.belief_state)
end

function filter_variance(filter::SingleFilter)
    return var([particle.x for particle in filter.particles]) + var([particle.y for particle in filter.particles])
end

function filter_mean_θ(filter::SingleFilter)
    return mean([atan(particle.y, particle.x) for particle in filter.particles])
end

function belief_to_observation(belief)
    θs_and_variances = sort([
        (filter_mean_θ(filter), filter_variance(filter)) for filter in belief
    ])

    s = []
    for (θ, var) in θs_and_variances
        push!(s, θ)
        push!(s, var)
    end
    while length(s) < 2 * PARAMS["number_of_targets"]
        push!(s, 0.0)
    end
    return SVector{2 * PARAMS["number_of_targets"]}(s)
end

### transitions

function generate_s(
    pomdp::BeliefPOMDP, s::UpdaterState, a::FlatAction, rng::RNG
) where {RNG<:AbstractRNG}
    sp = generate_s(pomdp.underlying_pomdp, s.underlying_state, a, rng)
    o = generate_o(pomdp.underlying_pomdp, s.underlying_state, a, sp, rng)
    bp = POMDPs.update(pomdp.updater, s.belief_state, a, o)
    return UpdaterState(sp, bp)
end

### rewards

function POMDPs.reward(pomdp::BeliefPOMDP, s::UpdaterState, a)
    return POMDPs.reward(pomdp.underlying_pomdp, s.underlying_state, s.belief_state)
end

### visualizations 

function POMDPTools.render(pomdp::BeliefPOMDP, step::NamedTuple)
    return draw_the_world(step.s.underlying_state, step.s.belief_state, step.a, []) #empty o
end
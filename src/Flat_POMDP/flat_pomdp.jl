# Saccades - eyes scan slowly, cause invisible motorcycles

# TODO: add dwell time 10e-3 to 50e-3 for 10-50ms to action space.
# TODO: add beamwidth selection to action space
# TODO: add target size to target attributes, use snr math to illuminate targets, 0.01m^3 bird to 500m^3 747

@kwdef struct FlatPOMDP <: POMDP{FlatState,FlatAction,FlatObservation} # POMDP{State, Action, Observation}
    rng::AbstractRNG
    discount::Number
    number_of_targets::Int64
    beamwidth_rad::Number
    radar_min_range_meters::Number
    radar_max_range_meters::Number
    n_particles::Int64
    xy_min_meters::Number
    xy_max_meters::Number
    dwell_time_seconds::Number # ∈ [10ms,40ms] # from Jack
    target_velocity_max_meters_per_second::Number # rounded up f-22 top speed is 700m/s
    target_reappearing_distribution::Sampleable
end

function POMDPs.isterminal(pomdp::FlatPOMDP, s::FlatState)
    # terminate early if no targets are in range
    for target in s.targets
        if √(target.x^2 + target.y^2) <= pomdp.radar_max_range_meters
            return false
        end
    end
    return true
end

function POMDPs.gen(
    pomdp::FlatPOMDP, s::FlatState, a::FlatAction, rng::RNG
) where {RNG<:AbstractRNG}
    sp = generate_s(pomdp, s, a, rng)
    o = generate_o(pomdp, s, a, sp, rng)
    # r = reward(pomdp, s, a)
    r = 0.0
    return (sp=sp, o=o, r=r)
end

### discount

function POMDPs.discount(pomdp::FlatPOMDP)
    return pomdp.discount
end

### states

function initialize_random_target(p::FlatPOMDP, i::Integer)::Target
    x = rand(p.rng, Uniform(p.xy_min_meters, p.xy_max_meters))
    y = rand(p.rng, Uniform(p.xy_min_meters, p.xy_max_meters))

    while √(x^2 + y^2) > p.radar_max_range_meters
        x = rand(p.rng, Uniform(p.xy_min_meters, p.xy_max_meters))
        y = rand(p.rng, Uniform(p.xy_min_meters, p.xy_max_meters))
    end

    ẋ = rand(
        p.rng,
        Uniform(
            -p.target_velocity_max_meters_per_second,
            p.target_velocity_max_meters_per_second,
        ),
    )
    ẏ = rand(
        p.rng,
        Uniform(
            -p.target_velocity_max_meters_per_second,
            p.target_velocity_max_meters_per_second,
        ),
    )
    if i == 1 # Make first target visible at start
        return Target(i, 0.0, x, y, ẋ, ẏ)
    end
    random_time = rand(p.rng, p.target_reappearing_distribution)
    return Target(i, random_time, x, y, ẋ, ẏ)
end

function POMDPs.initialstate(pomdp::FlatPOMDP)
    initial_targets = [
        initialize_random_target(pomdp, i) for i in 1:(pomdp.number_of_targets)
    ]

    return Deterministic(FlatState(initial_targets))
end

### actions 

function POMDPs.actions(pomdp::FlatPOMDP)
    # return Uniform{Float32}(0, 1)
    return collect(range(0, 1; length=40)) #TODO: set length to something like 2xbw
end

# observations 

function action_to_rad(action::FlatAction)
    return action * 2π - π # atan returns ∈ [-π,π], so lets just play nice
end

function target_visible(target::Target)
    return target.appears_at_t >= 0.0
end

function target_in_visible_range(target::Target, min_range::Number, max_range::Number)
    return min_range <= √(target.x^2 + target.y^2) <= max_range
end

function probability_detection(
    target::Target, boresight_rad::Number, time::Number, beamwidth::Number
)
    θ = atan(target.y, target.x)
    r = √(target.x^2 + target.y^2)

    # Handle beam crossing the π to -π line.
    if θ < 0
        θ = θ + 2π
    end
    if boresight_rad < 0
        θ = θ - 2π
    end

    signal = 7e20 # solve for 0.5 = 1 - e^{-x * 0.1 / 100000^4}

    if abs(θ - boresight_rad) < beamwidth / 2
        return 1 - ℯ^-(signal * time / r^4)
        # pd = 1 - ℯ^-(signal * time / r^4)
        # @info pd
        # return pd
    elseif abs(θ - boresight_rad) < beamwidth / 2 * 3
        return 1 - ℯ^-(signal / 256 * time / r^4) #250 is roughly -24db
    end
    return 0.0
end

function measurement_noise_r(rng, r)
    # based on noise covariance matrix in BWW (they use (0.1km)^2)
    return r + rand(rng, Normal(0, 100.0))
end

function measurement_noise_θ(rng, θ)
    # should be 1/2 BW / SNR (ref vkb)
    # for now, 0.1 based on noise covariance matrix in BWW
    return θ + rand(rng, Normal(0, 0.1 * 2π / 360))
end

function measurement_noise_v(rng, v)
    # doppler resolution 
    # ḋ = Normal(0.0, 6.0) # Some percent of range, really comes from PRF. Let's say PRF of 2khz, 100samples, v/λ, if λ is 50cm, so 6m/s.
    return v + rand(rng, Normal(0, 6.0))
end

function perfect_target_observation(target)
    # Sensor is at origin so this is all quite simple

    observed_r = √(target.x^2 + target.y^2)
    observed_θ = atan(target.y, target.x)

    target_local_θ = atan(target.ẏ, target.ẋ)
    target_local_v = √(target.ẋ^2 + target.ẏ^2)
    observed_v = cos(target_local_θ - observed_θ) * target_local_v

    return TargetObservation(target.id, observed_r, observed_θ, observed_v)
end

function noisy_target_observation(rng, target)
    perfect = perfect_target_observation(target)
    return TargetObservation(
        perfect.id,
        measurement_noise_r(rng, perfect.r),
        measurement_noise_θ(rng, perfect.θ),
        measurement_noise_v(rng, perfect.v),
    )
end

function illumination_observation(pomdp::FlatPOMDP, action::FlatAction, state::FlatState)

    # TODO: rare random detections

    observed_targets = filter(
        target ->
            target_visible(target) &&
                target_in_visible_range(
                    target, pomdp.radar_min_range_meters, pomdp.radar_max_range_meters
                ) &&
                # target_in_beam(target, action, pomdp.beamwidth_rad) &&
                probability_detection(
                    target,
                    action_to_rad(action),
                    pomdp.dwell_time_seconds, #TODO: put time in action
                    pomdp.beamwidth_rad,
                ) > rand(pomdp.rng),
        state.targets,
    )

    return [noisy_target_observation(pomdp.rng, target) for target in observed_targets]
end

function generate_o(
    pomdp::FlatPOMDP, s::FlatState, action::FlatAction, sp::FlatState, rng::AbstractRNG
)
    if isterminal(pomdp, sp)
        return TargetObservation[]
    end

    # Remember you make the observation at sp NOT s
    return illumination_observation(pomdp, action, sp)
end

# transitions 

function update_target(target::Target, pomdp::FlatPOMDP)
    if √(target.x^2 + target.y^2) > pomdp.radar_max_range_meters
        # If target has flown out of range, initialize a new target
        return initialize_random_target(pomdp, target.id)
    end
    return Target(
        target.id,
        target.appears_at_t + pomdp.dwell_time_seconds,
        target.x + target.ẋ * pomdp.dwell_time_seconds,
        target.y + target.ẏ * pomdp.dwell_time_seconds,
        target.ẋ,
        target.ẏ,
    )
end

function action_recency_index(a::FlatAction, recency::Array)
    return nothing
    return action_recency_index = Integer(
        min(length(recency), floor((a + 1 / length(recency)) * length(recency)))
    )
end

function generate_s(
    pomdp::FlatPOMDP, s::FlatState, a::FlatAction, rng::RNG
) where {RNG<:AbstractRNG}
    if isterminal(pomdp, s)
        # @error "No targets in frame, this should never happen"
        return s
    end
    # new_targets = SVector{pomdp.number_of_targets,Target}([
    #     update_target(target, pomdp) for target in s.targets
    # ])
    # return FlatState(new_targets)
    return FlatState([update_target(target, pomdp) for target in s.targets])
end

### rewards 

function POMDPs.reward(
    pomdp::FlatPOMDP,
    s::FlatState,
    b::MultiFilterBelief,
    sp::FlatState,
    bp::MultiFilterBelief,
)
    return POMDPs.reward(pomdp, sp, bp) - POMDPs.reward(pomdp, s, b)
end

function POMDPs.reward(pomdp::FlatPOMDP, s::FlatState, b::MultiFilterBelief)
    if isterminal(pomdp, s)
        return 0.0
        # Should never happen, does due to ugly inits and use of this in BetaZero action selection
        # Called as a result of 'empty' belief belief resulting in empty state.
        # an 'isempty(s)' early return is probably better 
    end
    return score_tracking(pomdp, s, b) + score_search(b)
end

function score_search(belief::MultiFilterBelief)
    return -sum(recency for recency in belief.azimuth_recency) / 6 # This is really action defined
end

function score_tracking(pomdp::FlatPOMDP, s::FlatState, b::MultiFilterBelief)
    score = 0.0
    tracked_targets = 0 # should be a filter and then 'length'
    for target in s.targets
        if target.appears_at_t >= 0 &&
            √(target.x^2 + target.y^2) <= pomdp.radar_max_range_meters
            filter_index = findfirst(x -> x.id == target.id, b.filters)
            # change to spread of the particle filter 
            if isnothing(filter_index)
                # # previously was a penalty, 
                # there's explicit search reward now, so no need to 'penalize' not spotting something
                # score += score_untracked_target(target)
            else
                tracked_targets = tracked_targets + 1
                score += score_tracked_target(target, b.filters[filter_index])
            end
        end
    end
    return score / tracked_targets # mean mean squared error 
end

function score_tracked_target(target, filter)
    belief = [[particle.x, particle.y] for particle in filter.particles]
    # MSE, so units are all meter^2
    return max(0, 1 - sum(([target.x, target.y] - mean(belief)) .^ 2 + var(belief)))
end

# function score_untracked_target(target)
#     return 0.0
#     # return -5e8
# end

# function expected_value(
#     pomdp::FlatPOMDP, s::FlatState, a::FlatAction, b::MultiFilterBelief, rng::RNG
# ) where {RNG<:AbstractRNG}
#     sp = generate_s(pomdp, s, a, rng)
#
#     return 3
# end

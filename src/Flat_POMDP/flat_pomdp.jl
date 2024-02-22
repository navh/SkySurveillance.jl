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
    r = reward(pomdp, s, a, sp)
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
    initial_state = FlatState(initial_targets)
    return Deterministic(initial_state)
end

### actions 

function POMDPs.actions(pomdp::FlatPOMDP)
    return Uniform{Float32}(0, 1)
end

# observations 

function action_to_rad(action::FlatAction)
    return action * 2π - π # atan returns ∈ [-π,π], so lets just play nice
end

function target_visible(target::Target)
    return target.appears_at_t >= 0.0
end

function target_in_range(target::Target, max_range::Number)
    return √(target.x^2 + target.y^2) < max_range
end

function target_in_beam(target::Target, action::FlatAction, beamwidth::Number)
    target_θ = atan(target.y, target.x)

    # Handle beam crossing the π to -π line.
    if target_θ < 0
        target_θ = target_θ + 2π
    end
    if action_to_rad(action) < 0
        target_θ = target_θ - 2π
    end
    return abs(target_θ - action_to_rad(action)) < beamwidth / 2
end

function target_observation(target, rng)
    # Sensor is at origin so this is all quite simple

    dr = Normal(0, 5) # range resolution
    dθ = Normal(0, 1 * 2π / 360) # should be 1/2 BW / SNR (talk to sunila about vkb)
    dv = Normal(0, 0.5) # doppler resolution

    observed_r = √(target.x^2 + target.y^2) + rand(rng, dr)
    observed_θ = atan(target.y, target.x) + rand(rng, dθ)
    target_local_θ = atan(target.ẏ, target.ẋ)
    target_local_v = √(target.ẋ^2 + target.ẏ^2)
    # TODO: add diagram to README showing how observed_v math works
    observed_v = cos(target_local_θ - observed_θ) * target_local_v + rand(rng, dv)

    return TargetObservation(target.id, observed_r, observed_θ, observed_v)
end

function illumination_observation(pomdp::FlatPOMDP, action::FlatAction, state::FlatState)

    # 1 - e**-SNR
    # SNR = POWER * t / (BW r**4)
    # work out the odds that the beam has hit the target
    # for now, hit = 1

    # TODO: rare random detections

    observed_targets = filter(
        target ->
            target_visible(target) &&
                target_in_range(target, pomdp.radar_max_range_meters) &&
                target_in_beam(target, action, pomdp.beamwidth_rad),
        state.targets,
    )

    return [target_observation(t, pomdp.rng) for t in observed_targets]
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
    if target_in_range(target, pomdp.radar_max_range_meters)
        return Target(
            target.id,
            target.appears_at_t + pomdp.dwell_time_seconds,
            target.x + target.ẋ * pomdp.dwell_time_seconds,
            target.y + target.ẏ * pomdp.dwell_time_seconds,
            target.ẋ,
            target.ẏ,
        )
    end
    return initialize_random_target(pomdp, target.id)
end

function generate_s(
    pomdp::FlatPOMDP, s::FlatState, a::FlatAction, rng::RNG
) where {RNG<:AbstractRNG}
    if isterminal(pomdp, s)
        @error "No targets in frame, this should never happen"
        return s
    end
    new_targets = SVector{pomdp.number_of_targets,Target}([
        update_target(target, pomdp) for target in s.targets
    ])
    return FlatState(new_targets)
end

### rewards 

# function POMDPs.reward(pomdp::FlatPOMDP, s::FlatState, a::FlatAction, sp::FlatState)
#     if isterminal(pomdp, s)
#         return 0
#     end
#
#     # sum of target reward?
#     # pomdp.sum_targets_observed
#     # I really still feel that this should be the reward distance reduction
#     # why have I done this to myself. why can't I just access b in here.
#     return 0
# end

function POMDPs.reward(pomdp::FlatPOMDP, s::FlatState, b::MultiFilterBelief)
    if isterminal(pomdp, s)
        return 0.0 # Should never happen? 
    end

    score = 0.0

    for target in s.targets
        if target.appears_at_t >= 0 &&
            √(target.x^2 + target.y^2) <= pomdp.radar_max_range_meters
            filter_index = findfirst(x -> x.id == target.id, b)
            # change to spread of the particle filter 
            if filter_index === nothing
                score += score_untracked_target(target)
            else
                score += score_tracked_target(target, b[filter_index])
            end
        end
    end
    return score / length(s.targets)
end

function score_tracked_target(target, filter)
    ### Attempt number 1: this explodes
    # 1e4 is experimentally roughly the worst a filter gets until it just flies off the handle
    # Basically RMS 
    # return 1 -
    #        (
    #     sum([
    #         √((target.x - particle.x)^2 + (target.y - particle.y)^2) for
    #         particle in filter.particles
    #     ]) / length(filter.particles)
    # ) / 8e4

    μ_x = mean([particle.x for particle in filter.particles])
    σ_x = std([particle.x for particle in filter.particles])

    μ_y = mean([particle.y for particle in filter.particles])
    σ_y = std([particle.y for particle in filter.particles])

    # return min(1.0, 1e7 * pdf(Normal(μ_x, σ_x), target.x) * pdf(Normal(μ_y, σ_y), target.y))
    return -((target.x - μ_x)^2 + σ_x^2 + (target.y - μ_y)^2 + σ_y^2)
end

function score_untracked_target(target)
    # return max(0.0, target.appears_at_t * 1e3)
    return -5e8
end

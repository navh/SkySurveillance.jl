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
    initial_state = FlatState(initial_targets)
    return Deterministic(initial_state)
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

function measurement_noise_r(rng, r)
    # range resolution
    return r + rand(rng, Normal(0, 5)) * 10
end

function measurement_noise_θ(rng, θ)
    # should be 1/2 BW / SNR (talk to sunila about vkb)
    return θ + rand(rng, Normal(0, 0.1 * 2π / 360))
end

function measurement_noise_v(rng, v)
    # doppler resolution 
    return v + rand(rng, Normal(0, 0.5)) * 10
end

function target_observation(target, rng)
    # Sensor is at origin so this is all quite simple

    observed_r = measurement_noise_r(rng, √(target.x^2 + target.y^2))
    observed_θ = measurement_noise_θ(rng, atan(target.y, target.x))

    # TODO: add diagram to README showing how observed_v math works
    target_local_θ = atan(target.ẏ, target.ẋ)
    target_local_v = √(target.ẋ^2 + target.ẏ^2)
    observed_v = measurement_noise_v(rng, cos(target_local_θ - observed_θ) * target_local_v)

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
        # @error "No targets in frame, this should never happen"
        return s
    end
    # new_targets = SVector{pomdp.number_of_targets,Target}([
    #     update_target(target, pomdp) for target in s.targets
    # ])
    new_targets = [update_target(target, pomdp) for target in s.targets]
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
    # return min(1.0, 1e7 * pdf(Normal(μ_x, σ_x), target.x) * pdf(Normal(μ_y, σ_y), target.y))

    # return 1 -
    #        mean([sqrt((target.x - p.x)^2 + (target.y - p.y)^2) for p in filter.particles]) / 2000

    x_particles = [particle.x for particle in filter.particles]
    y_particles = [particle.y for particle in filter.particles]
    return max(
        0,
        1 -
        (
            (target.x - mean(x_particles))^2 +
            var(x_particles) +
            (target.y - mean(y_particles))^2 +
            var(y_particles)
        ) / 1e7,
    )
end

function score_untracked_target(target)
    return 0.0
    # return -5e8
end

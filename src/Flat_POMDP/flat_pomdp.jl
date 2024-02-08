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

function initialize_random_targets(p::FlatPOMDP)::FlatState
    xs = rand(
        p.rng, Uniform(p.xy_min_meters * 0.8, p.xy_max_meters * 0.8), p.number_of_targets
    )
    ys = rand(
        p.rng, Uniform(p.xy_min_meters * 0.8, p.xy_max_meters * 0.8), p.number_of_targets
    )
    x_velocities = rand(
        p.rng,
        Uniform(
            -p.target_velocity_max_meters_per_second,
            p.target_velocity_max_meters_per_second,
        ),
        p.number_of_targets,
    )
    y_velocities = rand(
        p.rng,
        Uniform(
            -p.target_velocity_max_meters_per_second,
            p.target_velocity_max_meters_per_second,
        ),
        p.number_of_targets,
    )
    random_times = rand(p.rng, p.target_reappearing_distribution, p.number_of_targets)
    random_times[1] = 0 # at least one visible at first step
    initial_targets = [
        Target(i, random_times[i], xs[i], ys[i], x_velocities[i], y_velocities[i]) for
        i in 1:(p.number_of_targets)
    ]
    return FlatState(initial_targets)
end

function POMDPs.initialstate(pomdp::FlatPOMDP)
    # Maybe needs to be some implicit distribution.
    # Maybe just deterministic based on an RNG.
    return Deterministic(initialize_random_targets(pomdp))
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
    return target.appears_at_t >= 0
end

function target_in_range(target::Target, max_range::Number)
    return √(target.x^2 + target.y^2) < max_range
end

function target_in_beam(target::Target, action::FlatAction, beamwidth::Number)
    target_θ = atan(target.y, target.x)
    # TODO: check this around -pi to pi transition
    return abs((target_θ - action_to_rad(action))) < beamwidth / 2 # TODO: tried to fix the 180 prob, unsure about new issues around 0-1 transition?
end

function target_observation(target)
    # Sensor is at origin so this is all quite simple

    # TODO: gaussian noise goes here, depends on SNR?
    # Add in some sort of dθ?
    # dr = Normal(0, 1e-8) # range resolution
    # dθ = Normal(0, 1e-8)
    # dv = Normal(0, 1e-8) # doppler resolution
    dr = Normal(0, 5) # range resolution
    dθ = Normal(0, 3 * π / 180)
    dv = Normal(0, 0.5) # doppler resolution

    observed_r = √(target.x^2 + target.y^2) + rand(dr)
    observed_θ = atan(target.y, target.x) + rand(dθ) # note the reversal
    target_local_θ = atan(target.ẏ, target.ẋ) # note the reversal
    target_local_v = √(target.ẋ^2 + target.ẏ^2)
    # TODO: add diagram to README showing how observed_v math works
    observed_v = cos(target_local_θ - observed_θ) * target_local_v + rand(dv)

    return TargetObservation(target.id, observed_r, observed_θ, observed_v)
end

function illumination_observation(pomdp::FlatPOMDP, action::FlatAction, state::FlatState)

    # 1 - e**-SNR
    # SNR = POWER * t / (BW r**4)
    # work out the odds that the beam has hit the target
    # for now, hit = 1
    # put this in target_spotted?

    # TODO: rare random detections
    # - previously this was implemented by just generating a full matrix 0-1
    # - then I added my very small "Odds of random detection"
    # - then I np.floor'd it so that they almost all went to zero
    # - then I used that as a mask on a random grid of velocities
    # - can't recycle the random or they'll all be max velocity

    observed_targets = filter(
        target ->
            target_visible(target) &&
                target_in_range(target, pomdp.radar_max_range_meters) &&
                target_in_beam(target, action, pomdp.beamwidth_rad),
        state.targets,
    )

    return [target_observation(t) for t in observed_targets]
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

function update_target(target::Target, time)
    return Target(
        target.id,
        target.appears_at_t + time,
        target.x + target.ẋ * time,
        target.y + target.ẏ * time,
        target.ẋ,
        target.ẏ,
    )
end

function generate_s(
    pomdp::FlatPOMDP, s::FlatState, a::FlatAction, rng::RNG
) where {RNG<:AbstractRNG}
    if isterminal(pomdp, s)
        return s
    end
    new_targets = SVector{pomdp.number_of_targets,Target}([
        update_target(target, pomdp.dwell_time_seconds) for target in s.targets
    ])
    return FlatState(new_targets)
end

### rewards 

function POMDPs.reward(pomdp::FlatPOMDP, s::FlatState, a::FlatAction, sp::FlatState)
    if isterminal(pomdp, s)
        return 0
    end

    # sum of target reward?
    # pomdp.sum_targets_observed
    # I really still feel that this should be the reward distance reduction
    # why have I done this to myself. why can't I just access b in here. 
    return 0
end

function POMDPs.reward(pomdp::FlatPOMDP, s::FlatState, b::MultiFilterBelief)
    # if isterminal(pomdp, s)
    #     return 0
    # end
    #
    # visible_targets = filter(target -> target.appears_at_t >= 0, s.targets)
    #
    # tracked_targets = filter(
    #     target -> target.id ∈ [filter.id for filter in b], visible_targets
    # )
    # untracked_targets = filter(
    #     target -> target.id ∉ [filter.id for filter in b], visible_targets
    # )
    #
    score = 0

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
    return -score # NOTE: Negation of the score
end

function score_tracked_target(target, filter)
    # Basically RMS 
    return sum([
        √((target.x - particle.x)^2 + (target.y - particle.y)^2) for
        particle in filter.particles
    ]) / length(filter.particles)
end

function score_untracked_target(target)
    return target.appears_at_t * 1e3
end

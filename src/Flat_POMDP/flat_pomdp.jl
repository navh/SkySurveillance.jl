export FlatPOMDP, FlatState

# Saccades - eyes scan slowly, cause invisible motorcycles

# TODO: add dwell time 10e-3 to 50e-3 for 10-50ms to action space.
# TODO: add beamwidth selection to action space
# TODO: add target size to target attributes, use snr math to illuminate targets, 0.01m^3 bird to 500m^3 747

beamwidth_rad::Float32 = PARAMS["beamwidth_degrees"] * π / 360

XY_MAX_METERS::Float32 = PARAMS["radar_max_range_meters"]
XY_MIN_METERS::Float32 = -1 * PARAMS["radar_max_range_meters"]
XY_BIN_WIDTH::Float32 = (XY_MAX_METERS - XY_MIN_METERS) / PARAMS["xy_bins"]

DWELL_TIME_SECONDS::Float32 = PARAMS["dwell_time_seconds"] # ∈ [10ms,40ms] # from Jack
TARGET_VELOCITY_MAX_METERS_PER_SECOND::Float32 = PARAMS["target_velocity_max_meters_per_second"] # rounded up F-22 top speed is 700m/s

target_reappearing_distribution = Uniform(-50, 0)
target_reappearing_distribution = Uniform(-0.001, 0)

struct Target
    id::Int32
    appears_at_t::Float32
    x::Float32
    y::Float32
    ẋ::Float32
    ẏ::Float32
end

struct FlatState
    targets::SVector{PARAMS["number_of_targets"],Target}
end

struct TargetObservation
    id::Int32  # unique target id
    r::Float32 # range (meters)
    θ::Float32 # azimuth (radians)
    v::Float32 # radial velocity (meters/second)
end

FlatObservation = Vector{TargetObservation}

FlatAction = Float64 # Must be left 64 due to rand(uniform(f32,f32)) unwaveringly returning f64
# To play nice with MCTS should I just pick wedges?
# Eventually I'd like Tuple{Float32,Float32,Float32} # azimuth, beamwidth, dwell_time

@with_kw mutable struct FlatPOMDP <: POMDP{FlatState,FlatAction,FlatObservation} # POMDP{State, Action, Observation}
    rng::AbstractRNG
    discount::Float32 = 0.95 # was 1.0
end

function POMDPs.isterminal(pomdp::FlatPOMDP, s::FlatState)
    # terminate early if no targets are in range
    for target in s.targets
        if sqrt(target.x^2 + target.y^2) <= PARAMS["radar_max_range_meters"]
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

function initialize_random_targets(rng::RNG)::FlatState where {RNG<:AbstractRNG}
    xs = rand(rng, Uniform(XY_MIN_METERS, XY_MAX_METERS), PARAMS["number_of_targets"])
    ys = rand(rng, Uniform(XY_MIN_METERS, XY_MAX_METERS), PARAMS["number_of_targets"])
    x_velocities = rand(
        rng,
        Uniform(
            -TARGET_VELOCITY_MAX_METERS_PER_SECOND, TARGET_VELOCITY_MAX_METERS_PER_SECOND
        ),
        PARAMS["number_of_targets"],
    )
    y_velocities = rand(
        rng,
        Uniform(
            -TARGET_VELOCITY_MAX_METERS_PER_SECOND, TARGET_VELOCITY_MAX_METERS_PER_SECOND
        ),
        PARAMS["number_of_targets"],
    )
    random_times = rand(rng, target_reappearing_distribution, PARAMS["number_of_targets"])
    random_times[1] = 0 # Make at least one visible right first step
    initial_targets = [
        Target(i, random_times[i], xs[i], ys[i], x_velocities[i], y_velocities[i]) for
        i in 1:PARAMS["number_of_targets"]
    ]
    return FlatState(initial_targets)
end

function POMDPs.initialstate(pomdp::FlatPOMDP)
    # Maybe needs to be some implicit distribution.
    # Maybe just deterministic based on an RNG.
    return Deterministic(initialize_random_targets(pomdp.rng))
end

### actions 

function POMDPs.actions(pomdp::FlatPOMDP)
    return Uniform{Float32}(0, 1)
end

# observations 

function action_to_rad(action::FlatAction)
    # could also do -1 to 1 and just multiply by pi
    return action * 2π - π # atan returns ∈ [-π,π], so lets just play nice
end

function target_spotted(target::Target, action::FlatAction, beamwidth::Float32)
    if √(target.x^2 + target.y^2) > PARAMS["radar_max_range_meters"]
        return false
    end
    target_θ = atan(target.y, target.x)
    # TODO: check this around -pi to pi transition
    #return abs((target_θ - action_to_rad(action)) % π) < beamwidth # atan ∈ [-π,π] 
    return abs((target_θ - action_to_rad(action))) < beamwidth # TODO: tried to fix the 180 prob, unsure about new issues around 0-1 transition?
end

function target_observation(target)
    # Sensor is at origin so this is all quite simple
    r = √(target.x^2 + target.y^2)
    observed_θ = atan(target.y, target.x) # note the reversal
    target_local_θ = atan(target.ẏ, target.ẋ) # note the reversal
    target_local_v = √(target.ẋ^2 + target.ẏ^2)
    # TODO: add diagram to README showing how observed_v math works
    observed_v = cos(target_local_θ - observed_θ) * target_local_v

    # TODO: gaussian noise goes here, depends on SNR?

    return TargetObservation(target.id, r, observed_θ, observed_v)
end

function illumination_observation(action::FlatAction, state::FlatState)

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

    observations = TargetObservation[]

    observed_targets = filter(
        target -> target.appears_at_t >= 0 && target_spotted(target, action, beamwidth_rad),
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
    return illumination_observation(action, sp)
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
    new_targets = SVector{PARAMS["number_of_targets"],Target}([
        update_target(target, DWELL_TIME_SECONDS) for target in s.targets
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
    return 0
end

function POMDPs.reward(pomdp::FlatPOMDP, s::FlatState, b)
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
        if target.appears_at_t >= 0
            filter_index = findfirst(x -> x.id == target.id, b)
            # change to spread of the particle filter 
            if filter_index === nothing
                score += score_untracked_target(target)
            else
                score += score_tracked_target(target, b[filter_index])
            end
        end
    end
    return -score # NOTE: Negation of the scare
end

function score_tracked_target(target, filter)
    # Basically RMS 
    return sum([
        √((target.x - particle.x)^2 + (target.y - particle.y)^2) for
        particle in filter.particles
    ]) / length(filter.particles)
end

function score_untracked_target(target)
    return target.appears_at_t
end

# policies
function random_policy(pomdp, b)
    possible_actions = POMDPs.actions(pomdp, b)
    return rand(pomdp.rng, possible_actions)
end

# SAC 

# characteristics of the clouds
# mean and standard deviation, is particle cloud, variance
# give inputs as means and std deviations of clouds, locations of unseen targets.
# mean and standard deviations, of the clouds.
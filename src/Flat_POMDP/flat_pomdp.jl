export FlatPOMDP, FlatState, FlatBelief

# Saccades - eyes scan slowly, cause invisible motorcycles

# TODO: add dwell time 10e-3 to 50e-3 for 10-50ms to action space.
# TODO: add beamwidth selection to action space
# TODO: add target size to target attributes, use snr math to illuminate targets, 0.01m^3 bird to 500m^3 747

NUMBER_OF_TARGETS = 5

BEAMWIDTH = 10 * π / 360 # degree view as radians

RADAR_MIN_RANGE_METERS = 500  # "black zone" based on pulse width, can't RX while TX
RADAR_MAX_RANGE_METERS = 500_000  # up to 500km # based on height of radar/horizon

RANGE_BINS = 30  # Ravi had mentioned seeing a 600 bin example # Range Cells
AZIMUTH_BINS = 30 #  some factors of 360 are 18, 20, 24, 30, 36, 40, 45, 60, 72, 90, 120, 180

RANGE_SLICE_DEPTH = (RADAR_MAX_RANGE_METERS - RADAR_MIN_RANGE_METERS) / RANGE_BINS
AZIMUTH_SLICE_WIDTH = 2π / AZIMUTH_BINS

# no pizzaland
XY_MAX_METERS = RADAR_MAX_RANGE_METERS
XY_MIN_METERS = -1 * RADAR_MAX_RANGE_METERS
XY_BINS = 20
XY_BIN_WIDTH = (XY_MAX_METERS - XY_MIN_METERS) / XY_BINS

DWELL_TIME_SECONDS = 100e-3  # ∈ [10ms,40ms] # from Jack
TARGET_VELOCITY_MAX_METERS_PER_SECOND = 700  # rounded up F-22 top speed is 700m/s

#OBSERVATION_BUFFER_SIZE = 1000

CellType = Float64
#Cells= SMatrix{RANGE_BINS,AZIMUTH_BINS,CellType}
Cells = SMatrix{XY_BINS,XY_BINS,CellType}

struct Target
    x::Float64
    y::Float64
    x_velocity::Float64
    y_velocity::Float64
    t_since_observation::Float64 # Just for reward
end

struct FlatState
    targets::SVector{NUMBER_OF_TARGETS,Target}
end

function update_target(target::Target, time::Float64, observed::Bool)
    return Target(
        target.x + target.x_velocity * time,
        target.y + target.y_velocity * time,
        target.x_velocity,
        target.y_velocity,
        observed ? 0.0 : target.t_since_observation + time,
    )
end

struct TargetObservation
    r::Float64 # range (meters)
    θ::Float64 # azimuth (radians)
    v::Float64 # radial velocity (meters/second)
    t::Float64 # time since observation
end

FlatObservation = Vector{TargetObservation} # Ignore the below, I think this can just change in length 
# FlatObservation = SVector{NUMBER_OF_TARGETS,TargetObservation}
# I don't think this reveals the total number of targets in any meaningful way
# It ensures that on any observation I can *at least* fit them all

FlatAction = Float64
# To play nice with MCTS should I just pick wedges?
# Eventually I'd like Tuple{Float64,Float64,Float64} # azimuth, beamwidth, dwell_time
# But I'm already having dimensionality nightmares, so this will have to do for now.

struct FlatBelief
    #observations_buffer::SVector{OBSERVATION_BUFFER_SIZE,TargetObservation}
    occupancy_grid::Cells
end

@with_kw mutable struct FlatPOMDP <: POMDP{FlatState,FlatAction,FlatObservation} # POMDP{State, Action, Observation}
    rng::AbstractRNG
    discount::Float64 = 0.95 # was 1.0
end

function POMDPs.isterminal(pomdp::FlatPOMDP, s::FlatState)
    # terminate early if no targets are in range
    for target in s.targets
        if sqrt(target.x^2 + target.y^2) <= RADAR_MAX_RANGE_METERS
            return false
        end
    end
    return true
end

# # I've got no idea how to implement this, I don't think I need it
# function POMDPs.isterminal(pomdp::FlatPOMDP, b::FlatBelief)
#      return isterminal(pomdp, rand(pomdp.rng, pomdp, b))
# end

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
    xs = rand(rng, Uniform(XY_MIN_METERS, XY_MAX_METERS), NUMBER_OF_TARGETS)
    ys = rand(rng, Uniform(XY_MIN_METERS, XY_MAX_METERS), NUMBER_OF_TARGETS)
    x_velocities = rand(
        rng,
        Uniform(
            -TARGET_VELOCITY_MAX_METERS_PER_SECOND, TARGET_VELOCITY_MAX_METERS_PER_SECOND
        ),
        NUMBER_OF_TARGETS,
    )
    y_velocities = rand(
        rng,
        Uniform(
            -TARGET_VELOCITY_MAX_METERS_PER_SECOND, TARGET_VELOCITY_MAX_METERS_PER_SECOND
        ),
        NUMBER_OF_TARGETS,
    )
    initial_targets = [
        Target(xs[i], ys[i], x_velocities[i], y_velocities[i], 0) for
        i in 1:NUMBER_OF_TARGETS
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
    return Uniform(0, 1) # This feels wrong... I guess it's used for sampling?
end

"""
    POMDPs.action(p::RandomPolicy, b::FlatBelief)

A selection [0,1] representing where to look around a circle.
0 is west, 0.25 is north, 0.5 is east, 0.75 is south, 1 is west again.
"""
function POMDPs.action(p::RandomPolicy, b::FlatBelief)
    possible_actions = POMDPs.actions(p.problem, b)
    return rand(p.problem.pomdp.rng, possible_actions)
end

# POMDPs.actionindex(POMDP::FlatPOMDP, a::FlatAction) = a # Not discrete

# observations 

function action_to_rad(action::FlatAction)
    # could also do -1 to 1 and just multiply by pi
    return action * 2π - π # atan returns ∈ [-π,π], so lets just play nice
end

function target_spotted(target::Target, action::FlatAction, beamwidth::Float64)
    target_θ = atan(target.y, target.x)
    # TODO: check this
    # (0.530473384623918, Main.SkySurveillance.TargetObservation[Main.SkySurveillance.TargetObservation(336190.8852956795, -2.998197791214231, -25.437627674086766, 0.0)])
    # (0.41165095808460195, Main.SkySurveillance.TargetObservation[])
    # (0.010539030976806085, Main.SkySurveillance.TargetObservation[Main.SkySurveillance.TargetObservation(336187.83836025285, -2.99801559599274, -25.34462901551724, 0.0)])
    # (0.3336810511625745, Main.SkySurveillance.TargetObservation[])
    #return abs((target_θ - action_to_rad(action)) % π) < beamwidth # atan ∈ [-π,π] 
    return abs((target_θ - action_to_rad(action))) < beamwidth # TODO: tried to fix the 180 prob, unsure about new issues around 0-1 transition?
end

function real_occupancy(s::FlatState)
    #occupancy = zeros(Cells)
    occupancy = zeros(CellType, XY_BINS, XY_BINS)
    for target in s.targets
        x_bin = ceil(
            Int64, (target.x - XY_MIN_METERS) / (XY_MAX_METERS - XY_MIN_METERS) * XY_BINS
        )
        y_bin = ceil(
            Int64, (target.y - XY_MIN_METERS) / (XY_MAX_METERS - XY_MIN_METERS) * XY_BINS
        )
        if 0 < x_bin <= XY_BINS && 0 < y_bin <= XY_BINS
            occupancy[x_bin, y_bin] = 1
        end
    end
    return SVector{length(Cells),Float32}(occupancy) #Hack to make it play nice with the model
end

function target_observation(target)
    # Sensor is at origin so this is all quite simple
    r = √(target.x^2 + target.y^2)
    observed_θ = atan(target.y, target.x) # note the reversal
    target_local_θ = atan(target.y_velocity, target.x_velocity) # note the reversal
    target_local_v = √(target.x_velocity^2 + target.y_velocity^2)
    # TODO: add diagram to README showing how observed_v math works
    observed_v = cos(target_local_θ - observed_θ) * target_local_v
    t = 0.0

    # TODO: gaussian noise goes here

    return TargetObservation(r, observed_θ, observed_v, t)
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

    for target in state.targets
        if target_spotted(target, action, BEAMWIDTH)
            push!(observations, target_observation(target))
        end
    end
    return observations
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
        target.x + target.x_velocity * time,
        target.y + target.y_velocity * time,
        target.x_velocity,
        target.y_velocity,
        target.t_since_observation + time,
    )
end

function generate_s(
    pomdp::FlatPOMDP, s::FlatState, a::FlatAction, rng::RNG
) where {RNG<:AbstractRNG}
    if isterminal(pomdp, s)
        return s
    end
    new_targets = SVector{NUMBER_OF_TARGETS,Target}([
        update_target(target, DWELL_TIME_SECONDS) for target in s.targets
    ])
    return FlatState(new_targets)
end

# # Not totally sure why I had to define this when I'm using 'gen'...
# # Turns out I do not
# function POMDPs.transition(pomdp::FlatPOMDP, s::FlatState, a::FlatAction)
#     return generate_s(pomdp, s, a, pomdp.rng)
# end

### rewards 

function belief_reward(pomdp::FlatPOMDP, b::FlatBelief, a::FlatAction, bp::FlatBelief)
    if isterminal(pomdp, b)
        return 0
    end

    # reward distance reduction
    return 0
end

function POMDPs.reward(pomdp::FlatPOMDP, s::FlatState, a::FlatAction, sp::FlatState)
    if isterminal(pomdp, s)
        return 0
    end

    # sum of target reward?
    # pomdp.sum_targets_observed
    # I really still feel that this should be the reward distance reduction
    return 0
end

function POMDPs.reward(pomdp::FlatPOMDP, s::FlatState, b::FlatBelief)
    if isterminal(pomdp, s)
        return 0
    end

    # sum of target reward?
    # pomdp.sum_targets_observed
    # I really still feel that this should be the reward distance reduction
    return 0
end

# policies
function random_policy(pomdp, b)
    possible_actions = POMDPs.actions(pomdp, b)
    return rand(pomdp.rng, possible_actions)
end

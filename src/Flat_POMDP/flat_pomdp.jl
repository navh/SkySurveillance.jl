export FlatPOMDP, FlatState, FlatBelief

# Saccades - eyes scan slowly, cause invisible motorcycles

# TODO: add dwell time 10e-3 to 50e-3 for 10-50ms to action space.
# TODO: add beamwidth selection to action space
# TODO: add target size to target attributes, use snr math to illuminate targets, 0.01m^3 bird to 500m^3 747

beamwidth_rad::Float32 = PARAMS["beamwidth_degrees"] * π / 360

# RANGE_BINS = 30  # Ravi had mentioned seeing a 600 bin example # Range Cells
# AZIMUTH_BINS = 30 #  some factors of 360 are 18, 20, 24, 30, 36, 40, 45, 60, 72, 90, 120, 180

# RANGE_SLICE_DEPTH = (PARAMS["radar_max_range_meters"] - PARAMS["radar_min_range_meters"]) / RANGE_BINS
# AZIMUTH_SLICE_WIDTH = 2π / AZIMUTH_BINS

XY_MAX_METERS::Float32 = PARAMS["radar_max_range_meters"]
XY_MIN_METERS::Float32 = -1 * PARAMS["radar_max_range_meters"]
XY_BIN_WIDTH::Float32 = (XY_MAX_METERS - XY_MIN_METERS) / PARAMS["xy_bins"]

DWELL_TIME_SECONDS::Float32 = PARAMS["dwell_time_seconds"] # ∈ [10ms,40ms] # from Jack
TARGET_VELOCITY_MAX_METERS_PER_SECOND::Float32 = PARAMS["target_velocity_max_meters_per_second"] # rounded up F-22 top speed is 700m/s

#OBSERVATION_BUFFER_SIZE = 1000

CellType = Float32
#Cells= SMatrix{RANGE_BINS,AZIMUTH_BINS,CellType}
Cells = SMatrix{PARAMS["xy_bins"],PARAMS["xy_bins"],CellType}

struct Target
    id::Int32
    x::Float32
    y::Float32
    ẋ::Float32
    ẏ::Float32
end

struct FlatState
    targets::SVector{PARAMS["number_of_targets"],Target}
end

function update_target(target::Target, time::Float32, observed::Bool)
    return Target(
        target.id,
        target.x + target.ẋ * time,
        target.y + target.ẏ * time,
        target.ẋ,
        target.ẏ,
    )
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

struct FlatBelief
    occupancy_grid::Cells
end

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
    initial_targets = [
        Target(i, xs[i], ys[i], x_velocities[i], y_velocities[i]) for
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

"""
    POMDPs.action(p::RandomPolicy, b::FlatBelief)

A selection [0,1] representing where to look around a circle.
0 is west, 0.25 is north, 0.5 is east, 0.75 is south, 1 is west again.
"""
function POMDPs.action(p::RandomPolicy, b::FlatBelief)
    possible_actions = POMDPs.actions(p.problem, b)
    return rand(p.problem.pomdp.rng, possible_actions)
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
    # TODO: check this
    # (0.530473384623918, Main.SkySurveillance.TargetObservation[Main.SkySurveillance.TargetObservation(336190.8852956795, -2.998197791214231, -25.437627674086766, 0.0)])
    # (0.41165095808460195, Main.SkySurveillance.TargetObservation[])
    # (0.010539030976806085, Main.SkySurveillance.TargetObservation[Main.SkySurveillance.TargetObservation(336187.83836025285, -2.99801559599274, -25.34462901551724, 0.0)])
    # (0.3336810511625745, Main.SkySurveillance.TargetObservation[])
    #return abs((target_θ - action_to_rad(action)) % π) < beamwidth # atan ∈ [-π,π] 
    return abs((target_θ - action_to_rad(action))) < beamwidth # TODO: tried to fix the 180 prob, unsure about new issues around 0-1 transition?
end

# TODO: delete the following
# function real_occupancy(s::FlatState)
#     occupancy = zeros(CellType, PARAMS["xy_bins"], PARAMS["xy_bins"])
#     for target in s.targets
#         x_bin = ceil(
#             Int64,
#             (target.x - XY_MIN_METERS) / (XY_MAX_METERS - XY_MIN_METERS) *
#             PARAMS["xy_bins"],
#         )
#         y_bin = ceil(
#             Int64,
#             (target.y - XY_MIN_METERS) / (XY_MAX_METERS - XY_MIN_METERS) *
#             PARAMS["xy_bins"],
#         )
#         if 0 < x_bin <= PARAMS["xy_bins"] && 0 < y_bin <= PARAMS["xy_bins"]
#             occupancy[y_bin, x_bin] = 1
#         end
#     end
#     return occupancy
#     # return SVector{length(Cells),Float32}(occupancy) #Hack to make it play nice with the model
# end

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

    for target in state.targets
        if target_spotted(target, action, beamwidth_rad)
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
        target.id,
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

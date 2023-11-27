
# Functions in the distribution interface should be implemented if possible. 
# Implementing these functions will make the belief usable with many of the policies and planners in the POMDPs.jl ecosystem, 
# and will make it easy for others to convert between beliefs and to interpret what a belief means.

struct FlatBeliefUpdater{P<:POMDPs.POMDP} <: Updater
    pomdp::P
end

function POMDPs.initialize_belief(updater::FlatBeliefUpdater, d)
    return initial_belief_state(updater.pomdp, updater.pomdp.rng)
end

# # just always start empty.
# function POMDPs.initialize_belief(
#     updater::FlatBeliefUpdater, d, rng::RNG
# ) where {RNG<:AbstractRNG}
#     return initial_belief_state(updater.pomdp, rng)
# end

function initial_belief_state(pomdp::FlatPOMDP, rng::RNG) where {RNG<:AbstractRNG}
    observation_buffer = @SVector TargetObservation[]
    occupancy_grid = zeros(Cells) # square dimensions both bin count
    return FlatBelief(observation_buffer, occupancy_grid)
end

# function Base.rand(rng::AbstractRNG, pomdp::FlatPOMDP, b::FlatBelief)
#
#     empty_buffer = zeros(OBSERVATION_BUFFER_SIZE)
#     SVector{OBSERVATION_BUFFER_SIZE, TargetObservation} 
#     occupancy_grid::RangeAzimuthCells 
#     return FlatBelief(empty_buffer, occupancy_grid)
# end

function POMDPs.update(
    updater::FlatBeliefUpdater, b::FlatBelief, a::FlatAction, o::FlatObservation
)
    return update_belief(updater.pomdp, b, a, o, updater.pomdp.rng)
end

function update_belief(
    pomdp::P, b::FlatBelief, a::FlatAction, o::FlatObservation, rng::RNG
) where {P<:POMDPs.POMDP,RNG<:AbstractRNG}
    if isterminal(pomdp, b)
        return b
    end

    #TODO: totally remove this circular buffer and replace it all with just RNN updates?

    buffer = TargetObservation[]

    # Head of buffer will be most recent observations
    for t in o
        push!(buffer, t)
    end

    # Remainder of buffer will be most recent observations accumulating time
    for t in first(b.observations_buffer, OBSERVATION_BUFFER_SIZE - length(o))
        push!(buffer, TargetObservation(t.r, t.θ, t.v, t.t + DWELL_TIME_SECONDS))
    end

    new_buffer = SVector{length(buffer)}(buffer) # changing length on init is annoying

    fig = fully_illuminated_guess(new_buffer)

    return FlatBelief(new_buffer, fig)
end

function fully_illuminated_guess(buffer::SVector{TargetObservation})::Cells
    return zeros(Cells) # TODO: some zaney RNN?
end

struct Target
    id::Int32
    appears_at_t::Float32
    x::Float32
    y::Float32
    ẋ::Float32
    ẏ::Float32
end

struct FlatState
    targets::Vector{Target}
end

struct TargetObservation
    id::UInt32 # unique target id
    r::Float32 # range (meters)
    θ::Float32 # azimuth (radians)
    v::Float32 # radial velocity (meters/second)
end

FlatObservation = Vector{TargetObservation}

FlatAction = Number
# FlatAction = Float64 # Must be Float64 due to rand(rng, uniform(f32,f32)) unwaveringly returning f64
# I guess now could be a 'number'
# Eventually I'd like Tuple{Float32,Float32,Float32} # azimuth, beamwidth, dwell_time

struct WeightedParticle
    x::Float32
    y::Float32
    ẋ::Float32
    ẏ::Float32
    w::Float32
end

struct SingleFilter
    id::Int32
    particles::Vector{WeightedParticle}
end

MultiFilterBelief = Array{SingleFilter}

struct UpdaterState
    underlying_state::FlatState
    past_action::FlatAction
    underlying_observation::FlatObservation
    belief_state::MultiFilterBelief
end

UpdaterObservation = Vector{Float32} # SVector{2 * PARAMS["number_of_targets"],Float32}

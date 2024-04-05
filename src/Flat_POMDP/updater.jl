function propagate_particle(rng::AbstractRNG, p::WeightedParticle, time::Number)
    ẍ = rand(rng, Normal(0.0, √(40^2)))
    ÿ = rand(rng, Normal(0.0, √(40^2)))
    return WeightedParticle(
        p.x + p.ẋ * time + ẍ / 2 * time^2,
        p.y + p.ẏ * time + ÿ / 2 * time^2,
        p.ẋ + ẍ * time,
        p.ẏ + ÿ * time,
        p.w,
    )
end

function filter_variance(filter::SingleFilter)
    return sum(var([[particle.x, particle.y] for particle in filter.particles]))
end

function filter_variance_below_max(filter::SingleFilter, max_variance)
    return filter_variance(filter) <= max_variance
end

function propagate_filter(rng::AbstractRNG, filter::SingleFilter, time::Number)
    return SingleFilter(
        filter.id,
        [propagate_particle(rng, particle, time) for particle in filter.particles],
    )
end

function reweight_particle(particle::WeightedParticle, obs_x, obs_y)
    # weight = 1 / √((particle.x - obs_x)^2 + (particle.y - obs_y)^2) # just rms of distance? 
    weight = 1 / ((particle.x - obs_x)^2 + (particle.y - obs_y)^2) # just rms of distance? 
    # weight = max(0, √((particle.x - obs_x)^2 + (particle.y - obs_y)^2))
    return WeightedParticle(particle.x, particle.y, particle.ẋ, particle.ẏ, weight)
end

function reweight_filter(filter::SingleFilter, obs)
    x = obs.r * cos(obs.θ)
    y = obs.r * sin(obs.θ)
    return SingleFilter(filter.id, [reweight_particle(p, x, y) for p in filter.particles])
end

function initialize_filter(rng::AbstractRNG, obs, n_particles::Int)
    # x = obs.r * cos(obs.θ)
    # y = obs.r * sin(obs.θ)
    # ẋ = obs.v * cos(obs.θ)
    # ẏ = obs.v * sin(obs.θ)
    x_max_tangential_velocity = abs(700 * sin(obs.θ)) # 700 should be set to the max velocity param from the underlying pomdp
    y_max_tangential_velocity = abs(700 * cos(obs.θ))
    x_undertainty_in_tangential_velocity = Uniform(
        -x_max_tangential_velocity, x_max_tangential_velocity
    )
    y_undertainty_in_tangential_velocity = Uniform(
        -y_max_tangential_velocity, y_max_tangential_velocity
    )
    particles = [
        WeightedParticle(
            cos(measurement_noise_θ(rng, obs.θ)) * measurement_noise_r(rng, obs.r),
            sin(measurement_noise_θ(rng, obs.θ)) * measurement_noise_r(rng, obs.r),
            cos(measurement_noise_θ(rng, obs.θ)) * measurement_noise_r(rng, obs.v) +
            rand(rng, x_undertainty_in_tangential_velocity),
            sin(measurement_noise_θ(rng, obs.θ)) * measurement_noise_r(rng, obs.v) +
            rand(rng, y_undertainty_in_tangential_velocity),
            1.0,
        ) for _ in 1:n_particles
    ]
    return SingleFilter(obs.id, particles)
end

function weight_sum(filter::SingleFilter)
    return sum([particle.w for particle in filter.particles])
end

function weighted_centre_of_mass_in_range(filter::SingleFilter, range::Number)
    # Note: This isn't aggressive enough, if we only get a few scans very close to an edge it'll still just diverge forever
    # Consider giving filters some 'time to live' where they kill themselves after N timesteps without seeing a target?
    # Or just if some number (say 20%?) swim out of range then kill the whole thing? 
    # It's really just the special "one scan close to and edge and the target leaves" case that really makes a mess.
    # Just use the true target
    w = weight_sum(filter)
    mean_x = sum([particle.x * particle.w for particle in filter.particles]) / w
    mean_y = sum([particle.y * particle.w for particle in filter.particles]) / w
    return √(mean_x^2 + mean_y^2) <= range
end

function low_variance_resampler(rng::RNG, filter::SingleFilter) where {RNG<:AbstractRNG}
    ps = Array{WeightedParticle}(undef, length(filter.particles))
    r = rand(rng) * weight_sum(filter) / length(filter.particles)
    i = 1
    c = filter.particles[i].w
    U = r
    for m in 1:length(filter.particles)
        while U > c && i < length(filter.particles)
            i += 1
            c += filter.particles[i].w
        end
        U += weight_sum(filter) / length(filter.particles)
        ps[m] = filter.particles[i]
    end
    return SingleFilter(filter.id, ps)
end

# TODO: these belief updater params are stored in the underlying pomdp in a really awkward way
@kwdef struct MultiFilterUpdater <: POMDPs.Updater
    rng::AbstractRNG
    dwell_time_seconds::Number
    n_particles_per_filter::Number
    max_range::Number
    max_variance::Number
    recency_bins::Integer
end

function POMDPs.initialize_belief(up::MultiFilterUpdater)
    return POMDPs.initialize_belief(up, nothing)
end

function POMDPs.initialize_belief(up::MultiFilterUpdater, _)
    # I think that I should pass in the state and do a sweep to find all the initial targets
    initial_filters = SingleFilter[]
    initial_recency = [0.0 for _ in 1:(up.recency_bins)]
    return MultiFilterBelief(initial_filters, initial_recency)
end

function update_filters(
    up::MultiFilterUpdater,
    belief_old::MultiFilterBelief,
    action::FlatAction,
    observation::FlatObservation,
)
    # 1) Prediction (or propagation) - each state particle is simulated forward one step in time

    # TODO: pack 'time' in with action?
    time = up.dwell_time_seconds

    propagated_belief = [
        propagate_filter(up.rng, filter, time) for filter in belief_old.filters
    ]

    # 2) Reweighting - an explicit measurement (observation) model is used to calculate a new weight
    ### 3 cases: observationless filters, observed filters, filterless observations 

    # 2.1) - Non-observed, do nothing
    no_observed_filters = filter(x -> x.id ∉ [o.id for o in observation], propagated_belief)

    # 2.2) - Observed Filters, reweight based on observation, drop empty filters
    observed_filters = filter(x -> x.id ∈ [o.id for o in observation], propagated_belief)

    reweighted_filters = [
        reweight_filter(
            filter, observation[findfirst(obs -> obs.id == filter.id, observation)]
        ) for filter in observed_filters
    ]

    # Resampling - a new collection of state particles is generated with particle. frequencies proportional to the new weights
    resampled_filters = [low_variance_resampler(up.rng, f) for f in reweighted_filters] # O(n) resampler, page 110 of Probabilistic Robotics by Thurn, Burgard, and Fox.
    # resampled_filters = []

    # 2.3) - Filterless observations, initialize a new filter
    new_observations = filter(o -> o.id ∉ [f.id for f in propagated_belief], observation)
    # new_observations = observation

    new_filters = [
        initialize_filter(up.rng, o, up.n_particles_per_filter) for o in new_observations
    ]

    all_filters = vcat(no_observed_filters, resampled_filters, new_filters)

    # Throw out filters with a centre of mass outside the visible range
    visible_filters = filter(
        f -> weighted_centre_of_mass_in_range(f, up.max_range), all_filters
    )

    return filter(f -> filter_variance_below_max(f, up.max_variance), visible_filters)
end

function update_recency(
    up::MultiFilterUpdater,
    belief_old::MultiFilterBelief,
    action::FlatAction,
    observation::FlatObservation,
)
    # This relies on a ∈[0,1]
    # TODO: dwell_time_seconds into action
    recency = [t + up.dwell_time_seconds for t in belief_old.azimuth_recency]

    action_recency_index = Integer(
        min(length(recency), floor((action + 1 / length(recency)) * length(recency)))
    )

    recency[action_recency_index] = 0.0
    return recency
end

# function POMDPs.update(up::MultiFilterUpdater, belief_old, action, observation)
function POMDPs.update(
    up::MultiFilterUpdater,
    belief_old::MultiFilterBelief,
    action::FlatAction,
    observation::FlatObservation,
)
    return MultiFilterBelief(
        update_filters(up, belief_old, action, observation),
        update_recency(up, belief_old, action, observation),
    )
end

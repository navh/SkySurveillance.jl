function propagate_particle(p::WeightedParticle, time::Number)
    return WeightedParticle(
        p.x + p.ẋ * time,
        p.y + p.ẏ * time,
        p.ẋ + rand(Normal(0.0, √(time * (40^2 + 40^2)))),
        p.ẏ + rand(Normal(0.0, √(time * (40^2 + 40^2)))),
        p.w,
    )
    # return WeightedParticle(
    #     p.x + p.ẋ * time + rand(Normal(0.0, 25.0)),
    #     p.y + p.ẏ * time + rand(Normal(0.0, 25.0)),
    #     p.ẋ + rand(Normal(0.0, 6.0)),
    #     p.ẏ + rand(Normal(0.0, 6.0)),
    #     p.w,
    # )
end

function propagate_filter(filter::SingleFilter, time::Number)
    return SingleFilter(
        filter.id,
        [propagate_particle(particle, time) for particle in filter.particles],
        filter.last_x,
        filter.last_y,
        filter.last_t + time,
    )
end

function reweight_particle(particle::WeightedParticle, obs_x, obs_y, obs_ẋ, obs_ẏ)
    # weight = (
    #     pdf(Normal(0.0, 25.0), particle.x - obs_x) +
    #     pdf(Normal(0.0, 25.0), particle.y - obs_y)
    #     # pdf(Normal(0.0, 25.0), particle.x - obs_x) +
    #     # pdf(Normal(0.0, 25.0), particle.y - obs_y) +
    #     # pdf(Normal(0.0, 6.0), particle.ẋ - obs_ẋ) +
    #     # pdf(Normal(0.0, 6.0), particle.ẏ - obs_ẏ)
    #     # Should I be adding on some points here for nailing the doppler velocity?
    #     # The more I think about this, the less that I think it should be related to the other distributions
    #     # Should this just be 1/(1+RMS) or something?
    # )

    weight = 1 / (1 + √((particle.x - obs_x)^2 + (particle.y - obs_y)^2)) # just rms of distance? 

    return WeightedParticle(particle.x, particle.y, particle.ẋ, particle.ẏ, weight)
end

function reweight_filter(filter::SingleFilter, obs)
    x = obs.r * cos(obs.θ)
    y = obs.r * sin(obs.θ)
    ẋ = (x - filter.last_x) / filter.last_t # TODO: should I be using v at all here?
    ẏ = (y - filter.last_y) / filter.last_t

    # Throw away particles not moving in the right direction.

    return SingleFilter(
        filter.id,
        [reweight_particle(particle, x, y, ẋ, ẏ) for particle in filter.particles],
        x,
        y,
        0.0,
    )
end

function initialize_filter(obs, n_particles::Int)
    x = obs.r * cos(obs.θ)
    y = obs.r * sin(obs.θ)
    ẋ = obs.v * cos(obs.θ)
    ẏ = obs.v * sin(obs.θ)
    d = Normal(0.0, 25.0) # TODO: paramaterize at least σ? Ask ravi for a good theta? different d for r and v? Really comes from bandwidth, 3mhz, c/2b? +/- 25m. 1ghz radar, that would be 0.3% bandwidth.
    ḋ = Normal(0.0, 6.0) # Some percent of range, really comes from PRF. Let's say PRF of 2khz, 100samples, v/λ, if λ is 50cm, so 6m/s.
    x_max_tangential_velocity = abs(700 * sin(obs.θ))
    y_max_tangential_velocity = abs(700 * cos(obs.θ))
    x_undertainty_in_tangential_velocity = Uniform(
        -x_max_tangential_velocity, x_max_tangential_velocity
    )
    y_undertainty_in_tangential_velocity = Uniform(
        -y_max_tangential_velocity, y_max_tangential_velocity
    )
    particles = [
        WeightedParticle(
            x + rand(d),
            y + rand(d),
            ẋ + rand(ḋ) + rand(x_undertainty_in_tangential_velocity),
            ẏ + rand(ḋ) + rand(y_undertainty_in_tangential_velocity),
            1.0,
        ) for _ in 1:n_particles
    ]
    return SingleFilter(obs.id, particles, x, y, 0.0)
end

function weight_sum(filter::SingleFilter)
    return sum([particle.w for particle in filter.particles])
end

function weighted_centre_of_mass_in_range(filter::SingleFilter, range::Number)
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
    return SingleFilter(filter.id, ps, filter.last_x, filter.last_y, filter.last_t)
end

@kwdef struct MultiFilterUpdater <: POMDPs.Updater
    rng::AbstractRNG
    dwell_time_seconds::Number
    n_particles_per_filter::Number
    max_range::Number
end

function POMDPs.initialize_belief(up::MultiFilterUpdater, d)
    return SingleFilter[]
end

function POMDPs.update(up::MultiFilterUpdater, belief_old, action, observation)
    # 1) Prediction (or propagation) - each state particle is simulated forward one step in time

    # TODO: pack 'time' in with action?
    time = up.dwell_time_seconds

    propagated_belief = [propagate_filter(filter, time) for filter in belief_old]

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

    # 2.3) - Filterless observations, initialize a new filter
    new_observations = filter(o -> o.id ∉ [f.id for f in propagated_belief], observation)

    new_filters = [
        initialize_filter(o, up.n_particles_per_filter) for o in new_observations
    ]

    all_filters = vcat(no_observed_filters, resampled_filters, new_filters)

    # Throw out filters with a centre of mass outside the visible range
    return filter(f -> weighted_centre_of_mass_in_range(f, up.max_range), all_filters)
end

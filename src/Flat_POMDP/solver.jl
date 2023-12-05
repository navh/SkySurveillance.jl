# Note: fig from "fully illuminated guess", nothing to do with FIB, I'll probably rename due to similarity

struct FigFilterSolver <: Solver end

struct WeightedParticle
    x::Float32
    y::Float32
    ẋ::Float32
    ẏ::Float32
    w::Float32
end

struct SingleFilter
    id::Int32
    # particles::SVector{PARAMS["n_particles"],WeightedParticle}
    particles::Vector{WeightedParticle}
    last_x::Float32
    last_y::Float32
    last_t::Float32
end

function propagate_particle(p::WeightedParticle, time::Float64)
    return WeightedParticle(
        p.x + p.ẋ * time + rand(Normal(0.0, 200.0)),
        p.y + p.ẏ * time + rand(Normal(0.0, 200.0)),
        p.ẋ + rand(Normal(0.0, 16.0)), # Every time step... this is a LOT of noise
        p.ẏ + rand(Normal(0.0, 16.0)),
        p.w,
    )
end

function propagate_filter(filter::SingleFilter, time::Float64)
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
    #     #TODO: grab distributions from below or global these or something
    #     pdf(Normal(0.0, 25.0), particle.x - obs_x) +
    #     pdf(Normal(0.0, 25.0), particle.y - obs_y) +
    #     pdf(Normal(0.0, 6.0), particle.ẋ - obs_ẋ) +
    #     pdf(Normal(0.0, 6.0), particle.ẏ - obs_ẏ)
    #     # Should I be adding on some points here for nailing the doppler velocity?
    #     # The more I think about this, the less that I think it should be related to the other distributions
    #     # Should this just be 1/(1+RMS) or something?
    # )
    weight = 1 / (1 + √((particle.x - obs_x)^2 + (particle.y - obs_y)^2))# just rms of distance? 
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

function initialize_filter(obs)
    x = obs.r * cos(obs.θ)
    y = obs.r * sin(obs.θ)
    #ẋ = obs.v * cos(obs.θ) #TODO: does this check out? Should these maybe just be zero?
    #ẏ = obs.v * sin(obs.θ)
    ẋ = 0
    ẏ = 0
    d = Normal(0.0, 25.0) # TODO: paramaterize at least σ? Ask ravi for a good theta? different d for r and v? Really comes from bandwidth, 3mhz, c/2b? +/- 25m. 1ghz radar, that would be 0.3% bandwidth. Really low?.
    ḋ = Normal(0.0, 100.0) # Some percent of range, really comes from PRF. Let's say PRF of 2khz, 100samples, v/λ, if λ is 50cm, so 6m/s.
    particles = [
        WeightedParticle(x + rand(d), y + rand(d), ẋ + rand(ḋ), ẏ + rand(ḋ), 1.0) for
        _ in 1:PARAMS["n_particles"]
    ]
    return SingleFilter(obs.id, particles, x, y, 0.0)
end

function weight_sum(filter::SingleFilter)
    return sum([particle.w for particle in filter.particles])
end

function low_variance_resampler(filter::SingleFilter)
    ps = Array{WeightedParticle}(undef, PARAMS["n_particles"])
    r = rand(rng) * weight_sum(filter) / PARAMS["n_particles"]
    i = 1
    c = filter.particles[i].w
    U = r
    for m in 1:PARAMS["n_particles"]
        while U > c && i < length(filter.particles)
            i += 1
            c += filter.particles[i].w
        end
        U += weight_sum(filter) / PARAMS["n_particles"]
        ps[m] = filter.particles[i]
    end
    return SingleFilter(filter.id, ps, filter.last_x, filter.last_y, filter.last_t)
end

function POMDPs.solve(::FigFilterSolver, pomdp::POMDP)
    up = MultiFilterUpdater(pomdp.rng)
    return RandomPolicy(pomdp.rng, pomdp, up)
end

struct MultiFilterUpdater <: POMDPs.Updater
    rng::AbstractRNG
    # dynamics(x, u, rng) = x + u + randn(rng)
    # y_likelihood(x_previous, u, x, y) = pdf(Normal(), y - x)
    # model = ParticleFilterModel{Float32}(dynamics, y_likelihood)
    # pf = BootstrapFilter(model, 10)
end

function POMDPs.initialize_belief(up::MultiFilterUpdater, d)
    # TODO: think about random particles everywhere?
    return SingleFilter[] # Empty for now
end

function POMDPs.update(up::MultiFilterUpdater, belief_old, action, observation)
    # 1) Prediction (or propagation) - each state particle is simulated forward one step in time

    # TODO: pack 'time' in with action?
    time = PARAMS["dwell_time_seconds"]

    propagated_belief = [propagate_filter(filter, time) for filter in belief_old]

    # 2) Reweighting - an explicit measurement (observation) model is used to calculate a new weight
    ### 3 cases: observationless filters, filter with observation, filterless observations 

    # 2.1) - Non-observed, do nothing
    no_observed_filters = filter(x -> x.id ∉ [o.id for o in observation], propagated_belief)

    # 2.2) - Non-observed, reweight based on observation, drop empty filters
    observed_filters = filter(x -> x.id ∈ [o.id for o in observation], propagated_belief)

    reweighted_filters = [
        reweight_filter(
            filter, observation[findfirst(obs -> obs.id == filter.id, observation)]
        ) for filter in observed_filters
    ]

    # 2.3) - Filterless observations, initialize a new filter
    new_observations = filter(o -> o.id ∉ [f.id for f in propagated_belief], observation)

    new_filters = [initialize_filter(o) for o in new_observations]

    all_filters = vcat(no_observed_filters, reweighted_filters, new_filters)

    # 3) Resampling - a new collection of state particles is generated with particle. frequencies proportional to the new weights
    return [low_variance_resampler(f) for f in all_filters] # O(n) resampler, page 110 of Probabilistic Robotics by Thurn, Burgard, and Fox.
end
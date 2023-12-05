# Note: fig from "fully illuminated guess", nothing to do with FIB, I'll probably rename due to similarity

struct FigFilterSolver <: Solver end

struct WeightedParticle
    x::Float32
    y::Float32
    ẋ::Float32
    ẏ::Float32
    w::Float32
end

ParticleVector = SVector{PARAMS["n_particles"],WeightedParticle}

struct SingleFilter
    id::Int32
    particles::ParticleVector
end

struct MultiFilter = SingleFilter[] end

function propagate_particle(p::WeightedParticle, time::Float32)
    return WeightedParticle(p.x + p.ẋ * time, p.y + p.ẏ * time, p.ẋ, p.ẏ, p.w)
end

function propagate_filter(filter::SingleFilter, time::Float32)
    return [propagate_particle(particle, time) for particle in filter]
end

function reweight(filter::SingleFilter, observation)
    # TODO: implement
    return filter
end

function initialize_filter(obs)
    x = obs.r * cos(obs.θ)
    y = obs.r * sin(obs.θ)
    ẋ = obs.v * cos(obs.θ) #TODO: does this check out? Should these maybe just be zero?
    ẏ = obs.v * sin(obs.θ)
    d = Normal(; μ=0.0, σ=100.0) # TODO: paramaterize at least σ? Ask ravi for a good theta? different d for r and v? 
    particles = ParticleVector([
        WeightedParticle(x + rand(d), y + rand(d), ẋ + rand(d), ẏ + rand(d), 1.0) for
        _ in 1:PARAMS["n_particles"]
    ])
    return SingleFilter(obs.id, particles)
end

function low_variance_resampler(filter)
    # TODO: implement
    return filter
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
    reweighted_filters = [reweight(filter, observation) for filter in propagated_belief]

    # 2.3) - Filterless observations, initialize a new filter
    new_observations = filter(o -> o.id ∉ [f.id for f in propagated_belief], observation)

    new_filters = [initialize_filter(o) for o in new_observations]

    all_filters = vcat(no_observed_filters, reweighted_filters, new_filters)

    # 3) Resampling - a new collection of state particles is generated with particle. frequencies proportional to the new weights
    return [low_variance_resampler(f) for f in all_filters] # O(n) resampler, page 110 of Probabilistic Robotics by Thurn, Burgard, and Fox.
end
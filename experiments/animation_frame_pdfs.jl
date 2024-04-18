include("experiment_utils.jl") # Sets up the BMDP

# solver = SequentialSolver()
# solver = SingleSweepSolver()
solver = HighestVarianceSolver(0.5)

policy = solve(solver, bpomdp)

hr = HistoryRecorder(; max_steps=PARAMS["animation_steps"])
history = simulate(hr, bpomdp, policy)

function probability_detection(
    target::Target, boresight_rad::Number, time::Number, beamwidth::Number
)
    θ = atan(target.y, target.x)
    r = √(target.x^2 + target.y^2)

    # Handle beam crossing the π to -π line.
    if θ < 0
        θ = θ + 2π
    end
    if boresight_rad < 0
        θ = θ - 2π
    end

    signal = 7e20 # solve for 0.5 = 1 - e^{-x * 0.1 / 100000^4}

    if abs(θ - boresight_rad) < beamwidth / 2
        return 1 - ℯ^-(signal * time / r^4)
        # pd = 1 - ℯ^-(signal * time / r^4)
        # @info pd
        # return pd
    elseif abs(θ - boresight_rad) < beamwidth / 2 * 3
        return 1 - ℯ^-(signal / 256 * time / r^4) #250 is roughly -24db
    end
    return 0.0
end

function plot_var_by_a(step)
    plt = plot(;
        xlabel="Azimuth (rad)",
        ylabel="Expected Value",
        size=(4 * 72, 3 * 72),
        # fontfamily="Times New Roman",
        fontfamily="Arial",
        grid=false,
        fg_legend=:transparent,
    )

    # this incorrect beliefMDP is driving me up a wall

    azis = []
    vars = []
    recencies = []
    evs = []
    max_var = 0.0
    max_recency = 0.0
    for a in (-π):0.01:π
        var_accumulator = 0
        ev_accumulator = 0
        for filter in step.s.belief_state.filters
            target_variance = sum(
                var([[particle.x, particle.y] for particle in filter.particles])
            )
            # TODO: properly import this from flat_pomdp
            pd = probability_detection(
                Target(
                    0,
                    0,
                    mean([particle.x for particle in filter.particles]),
                    mean([particle.y for particle in filter.particles]),
                    0,
                    0,
                ),
                a,
                bpomdp.underlying_pomdp.dwell_time_seconds,
                bpomdp.underlying_pomdp.beamwidth_rad,
            )
            var_accumulator = var_accumulator + target_variance * pd
        end

        action_recency_index = Integer(
            min(
                length(step.s.belief_state.azimuth_recency),
                floor(
                    ((a + π) / (2π) + 1 / length(step.s.belief_state.azimuth_recency)) *
                    length(step.s.belief_state.azimuth_recency),
                ),
            ),
        )

        recency = step.s.belief_state.azimuth_recency[action_recency_index]

        push!(azis, a)
        push!(vars, var_accumulator)
        push!(recencies, recency)
        if var_accumulator > max_var
            max_var = var_accumulator
        end
        if recency > max_recency
            max_recency = recency
        end
    end

    vars = [var / max_var for var in vars]
    recencies = [recency / max_recency for recency in recencies]

    # probability_detection
    # MSE, so units are all meter^2

    plot!(plt, azis, vars; label="Variance")
    plot!(plt, azis, recencies; label="Recency")
    plot!(plt, [step.a * 2π - π]; seriestype="vline", color=:red, label="Action")

    return plt
end

function plot_recency_by_a(step)
    plt = plot(;
        xlabel="Azimuth (rad)",
        ylabel="Recency (s)",
        size=(4 * 72, 3 * 72),
        fontfamily="Times New Roman",
        grid=false,
    )

    azis = []
    recencies = []
    for a in (-π):0.1:π
        action_recency_index = Integer(
            min(
                length(step.s.belief_state.azimuth_recency),
                floor(
                    ((a + π) / (2π) + 1 / length(step.s.belief_state.azimuth_recency)) *
                    length(step.s.belief_state.azimuth_recency),
                ),
            ),
        )

        recency = step.s.belief_state.azimuth_recency[action_recency_index]

        push!(azis, a)
        push!(recencies, recency)
    end
    plot!(plt, azis, recencies; label="Recency")
    plot!(plt, [step.a * 2π - π]; seriestype="vline", color=:red, label="Action")

    return plt
end
#

# function plot_recency_by_a(step) end
@info "beginning rendering"
for step in eachstep(history)
    pdf(plot_var_by_a(step), dir_paths.figure_dir * "$(step.t)-var.pdf")
    pdf(plot_recency_by_a(step), dir_paths.figure_dir * "$(step.t)-recency.pdf")
    pdf(POMDPTools.render(bpomdp, step), dir_paths.figure_dir * "$(step.t)-frame.pdf")
end
@info dir_paths.figure_dir
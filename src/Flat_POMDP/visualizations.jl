DARKMODE = false

function circleShape(h, k, r)
    θ = LinRange(0, 2π, 500)
    return h .+ r * sin.(θ), k .+ r * cos.(θ)
end

function POMDPTools.render(pomdp::FlatPOMDP, step::NamedTuple)
    plt = plot(;
        axis=nothing,
        showaxis=false,
        background_color=DARKMODE ? :black : :white,
        # size=(1920, 1080),
        # size=(2 * 1920, 2 * 1080),
        size=(600, 600), # Only square resolution known to work with FFMPG
        xlims=(-pomdp.xy_max_meters, pomdp.xy_max_meters),
        ylims=(-pomdp.xy_max_meters, pomdp.xy_max_meters),
    )

    # Draw background circle
    plot!(
        plt,
        circleShape(0, 0, pomdp.radar_max_range_meters);
        seriestype=[:shape],
        lw=1,
        color=:black,
        linecolor=DARKMODE ? :green : :black,
        fillcolor=DARKMODE ? :grey : :grey,
        legend=false,
        fillalpha=1.0,
        aspect_ratio=1,
    )

    filter_colors = distinguishable_colors(
        pomdp.number_of_targets, [RGB(1, 1, 1), RGB(0, 0, 0)]; dropseed=true
    )

    # plot recency
    beamwidth_accumulator = -π
    for recency in step.b.azimuth_recency
        beam = Shape(
            [
                (0.0, 0.0)
                Plots.partialcircle(
                    beamwidth_accumulator,
                    beamwidth_accumulator + pomdp.beamwidth_rad,
                    100,
                    pomdp.radar_max_range_meters,
                )
                (0.0, 0.0)
            ],
        )
        plot!(
            plt,
            beam;
            fillcolor=DARKMODE ? :green : :white,
            linealpha=0.0,
            # linealpha=recency / pomdp.dwell_time_seconds / length(step.s.azimuth_recency),
            fillalpha=1 - (
                recency / 2 / pomdp.dwell_time_seconds / length(step.b.azimuth_recency)
            ),
        )
        beamwidth_accumulator = beamwidth_accumulator + pomdp.beamwidth_rad
    end

    # Plot particles
    for filter in step.b.filters
        plot!(
            plt,
            [(particle.x, particle.y) for particle in filter.particles];
            markercolor=filter_colors[filter.id],
            markeralpha=0.1,
            markerstrokewidth=0,
            seriestype=:scatter,
        )
    end

    # Draw action beam
    left = action_to_rad(step.a) - pomdp.beamwidth_rad / 2
    right = action_to_rad(step.a) + pomdp.beamwidth_rad / 2
    beam = Shape(
        [
            (0.0, 0.0)
            Plots.partialcircle(left, right, 100, pomdp.radar_max_range_meters)
            (0.0, 0.0)
        ],
    )
    # plot!(plt, beam; fillcolor=:red, linecolor=:white, fillalpha=0.1)
    plot!(plt, beam; fillcolor=DARKMODE ? :lime : :red, linealpha=0.0, fillalpha=0.3)

    # Draw -24db side lobes
    left24db = action_to_rad(step.a) - pomdp.beamwidth_rad / 2 * 3
    right24db = action_to_rad(step.a) + pomdp.beamwidth_rad / 2 * 3
    beam = Shape(
        [
            (0.0, 0.0)
            Plots.partialcircle(left24db, right24db, 100, pomdp.radar_max_range_meters)
            (0.0, 0.0)
        ],
    )
    plot!(plt, beam; fillcolor=DARKMODE ? :lime : :red, linealpha=0.0, fillalpha=0.3)

    # Draw visible circle
    plot!(
        plt,
        circleShape(0, 0, pomdp.radar_min_range_meters);
        seriestype=[:shape],
        lw=1,
        color=:black,
        linecolor=DARKMODE ? :green : :black,
        fillcolor=DARKMODE ? :black : :white,
        legend=false,
        fillalpha=1.0,
        aspect_ratio=1,
    )
    plot!(
        plt,
        circleShape(0, 0, pomdp.radar_max_range_meters);
        seriestype=[:shape],
        lw=1,
        color=:black,
        linecolor=DARKMODE ? :green : :black,
        legend=false,
        fillalpha=0.0,
        aspect_ratio=1,
    )

    # Add targets
    visible_targets = filter(target -> target.appears_at_t >= 0, step.s.targets)
    plot!(
        plt,
        [(target.x, target.y) for target in visible_targets];
        markercolor=DARKMODE ? :white : :black,
        markershape=:xcross,
        markersize=20,
        seriestype=:scatter,
    )

    # For debugging only 
    # invisible_targets = filter(target -> target.appears_at_t < 0, step.s.targets)
    # plot!(
    #     plt,
    #     [(target.x, target.y) for target in invisible_targets];
    #     markercolor=:gray,
    #     markershape=:xcross,
    #     markeralpha=[
    #         max(0.15, 1 + target.appears_at_t / 8) for target in invisible_targets
    #     ],
    #     markersize=10,
    #     seriestype=:scatter,
    # )

    # Plot observations
    if !isempty(step.o)
        plot!(
            plt,
            [(target.r * cos(target.θ), target.r * sin(target.θ)) for target in step.o];
            seriestype=:scatter,
            markershape=:xcross,
            markercolor=:red,
            markersize=25,
            markerstrokewidth=2,
            markerstrokecolor=:black,
        )
    end

    return plt
end

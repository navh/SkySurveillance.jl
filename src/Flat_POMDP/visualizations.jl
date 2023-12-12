function render(m::POMDP)
    return render(m, NamedTuple())
end

function POMDPTools.render(pomdp::FlatPOMDP, step::NamedTuple)
    return draw_the_world(step.s, step.b, step.a, step.o)
end

filter_colors = distinguishable_colors(
    PARAMS["n_particles"], [RGB(1, 1, 1), RGB(0, 0, 0)]; dropseed=true
)

function circleShape(h, k, r)
    θ = LinRange(0, 2π, 500)
    return h .+ r * sin.(θ), k .+ r * cos.(θ)
end

function draw_the_world(s, b, a, o)
    plt = plot(;
        axis=nothing,
        showaxis=false,
        size=(1920, 1080),
        xlims=(-XY_MAX_METERS, XY_MAX_METERS),
        ylims=(-XY_MAX_METERS, XY_MAX_METERS),
    )

    for filter in b
        plot!(
            plt,
            [(particle.x, particle.y) for particle in filter.particles];
            markercolor=filter_colors[filter.id],
            markeralpha=0.1,
            markerstrokewidth=0,
            seriestype=:scatter,
        )
    end

    # Draw visible circle
    plot!(
        plt,
        circleShape(0, 0, PARAMS["radar_max_range_meters"]);
        seriestype=[:shape],
        lw=0.5,
        color=:black,
        linecolor=:black,
        legend=false,
        fillalpha=0.0,
        aspect_ratio=1,
    )

    # Draw action beam
    left = action_to_rad(a) - beamwidth_rad
    right = action_to_rad(a) + beamwidth_rad
    beam = Shape(
        [
            (0.0, 0.0)
            Plots.partialcircle(left, right, 100, PARAMS["radar_max_range_meters"])
            (0.0, 0.0)
        ],
    )
    plot!(plt, beam; fillcolor=:red, fillalpha=0.5)

    # Plot observations
    if !isempty(o)
        plot!(
            plt,
            [(target.r * cos(target.θ), target.r * sin(target.θ)) for target in o];
            seriestype=:scatter,
            markershape=:xcross,
            markercolor=:red,
            markersize=25,
            markerstrokewidth=5,
            markerstrokecolor=:black,
        )
    end

    # Add targets
    visible_targets = filter(target -> target.appears_at_t >= 0, s.targets)
    invisible_targets = filter(target -> target.appears_at_t < 0, s.targets)
    plot!(
        plt,
        [(target.x, target.y) for target in visible_targets];
        markercolor=:blue,
        markershape=:xcross,
        markersize=10,
        seriestype=:scatter,
    )
    plot!(
        plt,
        [(target.x, target.y) for target in invisible_targets];
        markercolor=:gray,
        markershape=:xcross,
        markeralpha=[
            max(0.15, 1 + target.appears_at_t / 8) for target in invisible_targets
        ],
        markersize=10,
        seriestype=:scatter,
    )

    return plt
end

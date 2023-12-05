function render(m::POMDP)
    return render(m, NamedTuple())
end

function circleShape(h, k, r)
    θ = LinRange(0, 2π, 500)
    return h .+ r * sin.(θ), k .+ r * cos.(θ)
end

function POMDPTools.render(pomdp::FlatPOMDP, step)
    plt = plot(;
        axis=nothing,
        showaxis=false,
        size=(1920, 1080),
        xlims=(-XY_MAX_METERS, XY_MAX_METERS),
        ylims=(-XY_MAX_METERS, XY_MAX_METERS),
    )

    for filter in step.b
        plot!(
            plt,
            [(particle.x, particle.y) for particle in filter.particles];
            markeralpha=0.2,
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
    left = action_to_rad(step.a) - beamwidth_rad
    right = action_to_rad(step.a) + beamwidth_rad
    beam = Shape(
        [
            (0.0, 0.0)
            Plots.partialcircle(left, right, 100, PARAMS["radar_max_range_meters"])
            (0.0, 0.0)
        ],
    )
    plot!(plt, beam; fillcolor=:red, fillalpha=0.5)

    # Add targets
    plot!(
        plt,
        [(target.x, target.y) for target in step.s.targets];
        markercolor=:blue,
        markershape=:xcross,
        seriestype=:scatter,
    )

    # Plot observations
    if !isempty(step.o)
        plot!(
            plt,
            [(target.r * cos(target.θ), target.r * sin(target.θ)) for target in step.o];
            seriestype=:scatter,
            markershape=:xcross,
            markercolor=:red,
            markersize=10,
            markerstrokewidth=5,
        )
    end

    return plt
end
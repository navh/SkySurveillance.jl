function render(m::POMDP)
    return render(m, NamedTuple())
end

function circleShape(h, k, r)
    θ = LinRange(0, 2π, 500)
    return h .+ r * sin.(θ), k .+ r * cos.(θ)
end

function POMDPTools.render(pomdp::FlatPOMDP, step)
    plt = plot(;
        xlims=(-XY_MAX_METERS, XY_MAX_METERS), ylims=(-XY_MAX_METERS, XY_MAX_METERS)
    )

    #heatmap!(step.b; clims=(0.0, 1.0))
    xyticks = XY_MIN_METERS:((XY_MAX_METERS - XY_MIN_METERS) / XY_BINS):XY_MAX_METERS
    heatmap!(xyticks, xyticks, step.b; clims=(0.0, 1.0), c=:binary)

    # Draw visible circle
    plot!(
        plt,
        circleShape(0, 0, RADAR_MAX_RANGE_METERS);
        seriestype=[:shape],
        lw=0.5,
        color=:black,
        linecolor=:black,
        legend=false,
        fillalpha=0.0,
        aspect_ratio=1,
    )

    # Draw action beam
    left = action_to_rad(step.a) - BEAMWIDTH
    right = action_to_rad(step.a) + BEAMWIDTH
    beam = Shape(
        [
            (0.0, 0.0)
            Plots.partialcircle(left, right, 100, RADAR_MAX_RANGE_METERS)
            (0.0, 0.0)
        ]
    )
    plot!(plt, beam; fillcolor=:red, fillalpha=0.5)

    # Add targets
    plot!(
        plt,
        [(target.x, target.y) for target in step.s.targets];
        markercolor=:blue,
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
        )
    end

    return plt
end
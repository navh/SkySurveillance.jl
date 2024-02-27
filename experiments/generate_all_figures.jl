include("../src/SkySurveillance.jl")
using .SkySurveillance
using Dates: format, now
using Distributions: Uniform
using POMDPTools: Deterministic, HistoryRecorder, POMDPTools, eachstep
using POMDPs: mean, reward, simulate, solve
using Plots: @animate, Plots, mp4, pdf, plot, plot!, savefig;
using Random: Xoshiro
using TOML

function generate_all_figures()
    run_time = format(now(), "YYYYmmdd-HHMMSS-sss")
    @info "Run: $(run_time)"
    output_dir = "./out/$(run_time)/"
    dir_paths = (
        output_dir=output_dir,
        animation_dir=output_dir * "animations/",
        log_dir=output_dir * "logs/",
        figure_dir=output_dir * "figures/",
        model_dir=output_dir * "models/",
    )
    for dir_path in dir_paths
        mkpath(dir_path)
    end

    rng = Xoshiro(0)

    child_pomdp = FlatPOMDP(;
        rng=rng,
        discount=0.95,
        number_of_targets=1,
        beamwidth_rad=360 / 20 * π / 180,
        radar_min_range_meters=500,
        radar_max_range_meters=500_000,
        n_particles=200,
        xy_min_meters=-1 * 500_000,
        xy_max_meters=500_000,
        dwell_time_seconds=200e-3,
        target_velocity_max_meters_per_second=700,
        target_reappearing_distribution=Uniform(-1e-10, 0),
    )
    u = MultiFilterUpdater(
        child_pomdp.rng,
        child_pomdp.dwell_time_seconds,
        child_pomdp.n_particles,
        child_pomdp.radar_max_range_meters,
        1e8,
    )
    pomdp = BeliefPOMDP(child_pomdp.rng, child_pomdp, u)

    # solver_single_sweep = SingleSweepSolver()
    # solver_single_sweep = SequentialSolver()
    solver_single_sweep = HighestVarianceSolver()

    policy_single_sweep = solve(solver_single_sweep, pomdp)
    hr = HistoryRecorder(; max_steps=1000)
    four_sweeps = [simulate(hr, pomdp, policy_single_sweep) for _ in 1:4]
    plt = plot(;
        xlabel="Step",
        ylabel="Belief-Reward",
        # There's 72 points in an inch
        # In theory an ieee col is 3.5 inches wide
        # size=(3.5 * 72, 2.5 * 72),
        size=(7 * 72, 5 * 72),
        titlefont=("STIXTwoText"),
        legendfont=("STIXTwoText"),
        tickfont=("STIXTwoText"),
        guidefont=("STIXTwoText"),
        titlefontsize=6,
        legendfontsize=6,
        tickfontsize=6,
        guidefontsize=6,
    )
    for history in four_sweeps
        rewards = [step.r for step in history]
        plot!(plt, rewards; label="")
    end
    fig_path = dir_paths.figure_dir * "four_single_detections.pdf"
    @info fig_path
    savefig(plt, fig_path)

    # the 'clouds', maybe make better figures explaining particle filter
    # cloud_converged.pdf
    # cloud_diverging.pdf
    # compare.pdf
    # first_hit.pdf
    # second_hit.pdf
    # three.pdf

    return nothing
end

generate_all_figures()

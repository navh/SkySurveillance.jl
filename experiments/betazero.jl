include("experiment_utils.jl") # Sets up the BMDP

solver = BetaZeroSolver(;
    pomdp=pomdp,
    updater=up,
    params=BetaZeroParameters(; n_iterations=4, n_data_gen=50),
    nn_params=BetaZeroNetworkParameters(
        pomdp,
        up;
        training_epochs=50,
        n_samples=100_000,
        batchsize=1024,
        learning_rate=1e-4,
        λ_regularization=1e-5,
        use_dropout=true,
        p_dropout=0.2,
        use_prioritized_action_selection=false,
        device=cpu,
        optimizer=Flux.Adam, # Training optimizer (e.g., Adam, Descent, Nesterov)
        # use_epsilon_greedy=true, # Use epsilon-greedy exploration during action widening
        # ϵ_greedy=0.05,
    ),
    verbose=true,
    collect_metrics=true,
    plot_incremental_data_gen=false,
    mcts_solver=PUCTSolver(;
        n_iterations=100,
        k_action=2.0,
        alpha_action=0.25,
        k_state=2.0,
        alpha_state=0.1,
        enable_action_pw=true,
        enable_state_pw=true,
        counts_in_info=true, # Note, required for policy vector.
        final_criterion=MaxZQN(; zq=1, zn=1),
        exploration_constant=1.0,
    ),
)

policy = solve(solver, pomdp)
save_policy(policy, dir_paths.model_dir * "policy.bson")
save_solver(solver, dir_paths.model_dir * "solver.bson")
@info dir_paths.model_dir

"""
Use the raw policy head of the network to get the next action given a belief.
Based on: https://github.com/sisl/BetaZero.jl/blob/main/src/raw_network.jl
"""
mutable struct RawNetworkPolicy <: Policy
    pomdp::POMDP
    surrogate::Surrogate
end

function POMDPs.action(policy::RawNetworkPolicy, b)
    problem = policy.pomdp
    A = POMDPs.actions(problem)
    Ab = POMDPs.actions(problem, b)
    p = policy_lookup(policy.surrogate, b)

    # Match indices of (potentially) reduced belief-dependent action space to get correctly associated probabilities from the network
    if length(A) != length(Ab)
        idx = Vector{Int}(undef, length(Ab))
        for (i, a) in enumerate(A)
            for (j, ab) in enumerate(Ab)
                if a == ab
                    idx[j] = i
                    break
                end
            end
        end
        p = p[idx]
    end

    # pidx = sortperm(p)
    # # BetaZero.UnicodePlots.barplot(actions(pomdp)[pidx], _P[pidx]) |> display
    # BetaZero.UnicodePlots.barplot(Ab[pidx], p[pidx]) |> display

    exponentiate_policy = false
    if exponentiate_policy
        τ = 2
        p = normalize(p .^ τ, 1)
        return rand(SparseCat(Ab, p))
    else
        return Ab[argmax(p)]
    end
end

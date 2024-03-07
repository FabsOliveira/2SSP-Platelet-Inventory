# Uses Nesterov accelerated gradient descent. 1D demand policy by default.

using Plots
using LinearAlgebra
using Distributed
using Statistics
include("model_functions.jl")

# ISSUES: servicelevel-constraint causes infeasibility
# the same constraint creates problems with NIS policy
# maybe an additional cost could be introduced, if less than
# α % of the demand is satisfied, then problems with infeasibility
# would no longer exist...
function hospital_fun(
                instance::InstanceData,
                rho1_val = 50.0,
                rho2_val = 100.0,
                sens = 0,
                a_arg::Float64 = 0.0,
                b_arg::Float64 = 0.0) # tells if sensitivity analysis is performed

    Xi = length(instance.Xi) #20         # no of scenarios
    T = length(instance.T)  #14          # no of time periods (14 by default)
    M = length(instance.M)  #3           # shelf life of platelets
    P = 1 / Xi      # prob of each sccenario

    ins = instance

    N = 1500                      # max no of iterations
    Stot_all = zeros(N + 1, Xi) # store optimal order-up-to levels
    Sm_all = zeros(N + 1, Xi)   # store optimal minimum order amounts
    LB_scen = zeros(N + 1, Xi)  # scenario-wise lower bounds
    LB_all = zeros(N)       # store obtained lower bound values

    λ = zeros(N + 1, Xi)        # lagr multipliers (note that sum(λ) = 0 must hold)
    μ = zeros(N + 1, Xi)
    rho1 = 0.0                  # penalty params for quadr terms
    rho2 = 0.0                  # rho = 0 for first round
    S̄tot = zeros(N + 1)                  # for coupling constraints (default 0)
    S̄m   = zeros(N + 1)

    if sens == 2 # use given eta params for step size sensitivity analysis
        a = a_arg
        b = b_arg
    else # parameters for Nesterov algorithm
        η = 0.1
        a = 0.1
        b = 0
    end

            # η = 1 good when α = 0 - 0.1, β = 0.95
            # η = 0.1 good when α = 0.9, β = 0.8 (basic or nesterov), 1 when momentum
            # momentum gives good results in 20 iterations
    v_λ = zeros(Xi)             # weighted sums for...
    v_μ = zeros(Xi)             # previous gradients
    γ   = 0.9                   # decay parameter
    ε   = 0.1 ### make this smaller (0.01) later on to get accurate results!
    t_arr = zeros(N)

    t_max = 5000 # max allowed running time. May not be used.

    i = 0
    println("Stopping threshold ε: $(ε)\n")
    while i < N && (i == 0 || norm([S̄tot[i] .- Stot_all[i, :], S̄m[i] .- Sm_all[i, :]])^2 >= ε) && sum(t_arr) < t_max
        round_i = @timed begin
            if (i % 10 == 0 && i >= 1)
                resid = norm([S̄tot[i] .- Stot_all[i, :], S̄m[i] .- Sm_all[i, :]])^2
                println("Round $(i)\t\tResidual value $(resid)")
            end
            # in first round (i = 0), calculate init values
            # then iterate:
            # 1) solve optimal Stot, Sm for each scenario
            # 2) update S̄tot = mean(Stot_xi), S̄m = mean(Sm_xi)
            # or maximize z wrt S̄ when Stot, Sm known for all xi (same results)
            # 3) update λ and μ (many ways, subgr used here)

            λ_t = λ[i + 1, :] + γ * v_λ
            μ_t = μ[i + 1, :] + γ * v_μ

            res_arr = solve_parallel_model(ins, λ_t, μ_t, S̄tot[i + 1], S̄m[i + 1], rho1, rho2)
            for xi in 1:Xi
                if ins.policy == "1D"
                    Stot_all[i + 1, xi] = res_arr[xi]
                elseif ins.policy == "NIS"
                    Sm_all[i + 1, xi] = res_arr[xi]
                else
                    Stot_all[i + 1, xi] = res_arr[xi][1]
                    Sm_all[i + 1, xi] = res_arr[xi][2]
                end
            end

            # the lower bound is the Lagrangian dual value with given multipliers and augm. lagr. optimal Stot and sm
            # LB_all[i + 1] = solveLagrvalue(ins, λ[i + 1, :], μ[i + 1, :], S̄tot, S̄m) #, Stot_all[i + 1, :], Sm_all[i + 1, :])
            # or calculate values using maximum value of S̄tot and S̄m to ensure servicelevel
            if i % 5 == 0
                LB_all[i + 1] = solveLagrvalue(ins, λ[i + 1, :], μ[i + 1, :], S̄tot[i + 1], S̄m[i + 1])
            else
                LB_all[i + 1] = LB_all[i]
            end
            # update coupling parameters (same results when maximizing aug dual wrt these)
            if ins.policy == "1D" || ins.policy == "2D"
                S̄tot[i + 2] = mean(Stot_all[i + 1, :])
            end
            if ins.policy == "NIS" || ins.policy == "2D"
                S̄m[i + 2] = mean(Sm_all[i + 1, :])
            end
            # update penalty params (at least make them nonzero when i > 0)
            # rho = 1000 gives nice results when α, β = 0
            # for NIS rho2 = 1000 * (i + 1)
            rho1 = rho1_val * (i + 1)
            rho2 = rho2_val * (i + 1)
            # update multipliers

            if ins.policy == "1D" || ins.policy == "2D"
                λ[i + 2, :] = λ_t + a / (i + b + 1) * P * (S̄tot[i + 2] .- Stot_all[i + 1, :])
                v_λ = γ * v_λ + a / (i + b + 1) * P * (S̄tot[i + 2] .- Stot_all[i + 1, :])
            end
            if ins.policy == "NIS" || ins.policy == "2D"
                μ[i + 2, :] = μ_t + a / (i + b + 1) * P * (S̄m[i + 2] .- Sm_all[i + 1, :])
                v_μ = γ * v_μ + a / (i + b + 1) * P * (S̄m[i + 2] * .- Sm_all[i + 1, :])
            end

            i += 1
        end
        t_arr[i] = round_i[2]
    end
    println("\nIn total, $(i) iteration rounds were made.")
    if sens == 0
        return Stot_all[2:N + 1, :], Sm_all[2:N + 1, :], LB_all, λ[1:N, :], μ[1:N, :], i - 1, t_arr
    else
        return t_arr, LB_all, Stot_all[2:N + 1, :], Sm_all[2:N + 1, :] # if doing sensitivity analysis, just return LB values
    end

end

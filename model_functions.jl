# using Distributions
using Distributed
using JuMP
using Gurobi       
using LinearAlgebra

# nl_solver = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
gurobienv = Gurobi.Env()

# functions for augmented Lagrangian method
@everywhere begin
    function solve_parallel_scen(xi::Int, Id::InstanceData, lambda::Array{Float64, 1},
        mu::Array{Float64, 1}, Stot::Float64, Sm::Float64, rho1::Float64, rho2::Float64)
        if Id.policy == "1D"
            model = create_aug_model(Id, lambda, zeros(length(Id.Xi)), Stot, Sm, rho1, 0.0, xi)
            status = optimize!(model)
            return value(model[:Stot])
        elseif Id.policy == "NIS"
            model = create_aug_model(Id, zeros(length(Id.Xi)), mu, Stot, Sm, 0.0, rho2, xi)
            status = optimize!(model)
            return value(model[:Sm])
        else
            model = create_aug_model(Id, lambda, mu, Stot, Sm, rho1, rho2, xi)
            status = optimize!(model)
            return value(model[:Stot]), value(model[:Sm])
        end
    end
end

function solve_parallel_model(Id::InstanceData, λ::Array{Float64, 1}, μ::Array{Float64, 1},
        S̄tot::Float64, S̄m::Float64, rho1::Float64, rho2::Float64)
    # pmap solves the scenarios in parallel. The syntax: solving function plus parameters for each scenario
    # e.g. map(x -> λ, Id.Xi) returns a vector of size Id.Xi whose each element is λ
    results = pmap(solve_parallel_scen,
              Id.Xi,
              map(x -> Id, Id.Xi),
              map(x -> λ, Id.Xi),
              map(x -> μ, Id.Xi),
              map(x -> S̄tot, Id.Xi),
              map(x -> S̄m, Id.Xi),
              map(x -> rho1, Id.Xi),
              map(x -> rho2, Id.Xi))
    # results is an array of tuples, size Id.Xi x 1 or 2
    return results
end

function create_aug_model(Id::InstanceData, λ::Array{Float64, 1}, μ::Array{Float64, 1},
     S̄tot::Float64, S̄m::Float64, rho1::Float64, rho2::Float64, xi::Int64, solving_UB = false)
    # creates an augmented lagrangian dual for a single scenario

    model = Model(with_optimizer(Gurobi.Optimizer, gurobienv, OutputFlag = 0))
    # model = optimizer_with_attributes(() -> Gurobi.Optimizer(gurobienv), "OutputFlag" => 0)
    # model = Model(optimizer_with_attributes(Juniper.Optimizer, "nl_solver" => nl_solver, "log_levels" => [])) # no output pls
    if !solving_UB
        @variables model begin
            q[Id.T] >= 0#, Int                 # order quantity
            is[Id.T, Id.M] >= 0#, Int          # start inventory
            ie[Id.T, Id.M] >= 0#, Int          # end inventory
            a[Id.T, Id.M] >= 0#, Int           # no of units of age M used
            f[Id.T] >= 0#, Int                 # shortage (not dependent on the age)
            v[Id.T] >= 0#, Int                 # total inventory at end of t
            e[Id.T] >= 0#, Int                 # outdated
        end
        ### Added B as a variable to be optimised, if solving SVP
        if Id.solving_SVP
            @variable(model, B[Id.M] >= 0)
        end
    else # when solving UB, force all these to have only Int values
        @variables model begin
            q[Id.T] >= 0, Int                 # order quantity
            is[Id.T, Id.M] >= 0, Int          # start inventory
            ie[Id.T, Id.M] >= 0, Int          # end inventory
            a[Id.T, Id.M] >= 0, Int           # no of units of age M used
            f[Id.T] >= 0, Int                 # shortage (not dependent on the age)
            v[Id.T] >= 0, Int                 # total inventory at end of t
            e[Id.T] >= 0, Int                 # outdated
        end
        ### Added B as a variable to be optimised, if solving SVP
        if Id.solving_SVP
            @variable(model, B[Id.M] >= 0, Int)
        end
    end

    # add a few more variables & constraints depending on policy used
    if Id.policy == "NIS"
        @variable(model, Sm >= 0)           # amount to order in each round
        @constraint(model, NISorder[t=Id.T], q[t] == Sm); #order quantity is always Sm
    elseif Id.policy == "1D"
        @variables model begin
            Stot >= 0                       # order-up-to level
            b[Id.T], Bin                    # = 1 if there is shortage
        end
        #linearized constraints for order quantity
        @constraint(model, lin1[t=Id.T],    # amount to be ordered at least the amount of deficit
            q[t] >= Stot - sum(is[t,m] for m in Id.M[Id.L+1:end]));
        @constraint(model, lin2[t=Id.T],    # amount to be ordered not more than deficit (S̄ so that q >= 0 is possible)
            q[t] <= Stot - sum(is[t,m] for m in Id.M[Id.L+1:end]) + Id.S*(1-b[t]));
        @constraint(model, lin3[t=Id.T],    # if no deficit, don't order anything
            q[t] <= Id.S*b[t]);
    else    # dealing with 2D policy
        @variables model begin
            Stot >= 0
            Sm >= 0
            b[Id.T], Bin                    # = 1 if deficit > Sm
        end
        # linearized constraints for order quantity
        @constraint(model, linear1[t=Id.T], # amount to be ordered at least the amount of deficit
            q[t] >= Stot - sum(is[t,m] for m in Id.M[Id.L+1:end]));
        @constraint(model, linear2[t=Id.T], # always order at least Sm items of fresh platelets
            q[t] >= Sm);
        @constraint(model, linear3[t=Id.T], # amount to be ordered not more than deficit (S̄ so that q >= 0 is possible)
            q[t] <= Stot - sum(is[t,m] for m in Id.M[Id.L+1:end]) + Id.S*(1-b[t]));
        @constraint(model, linear4[t=Id.T], # amount to be ordered = Sm if stock is already full (then b = 0)
            q[t] <= Sm + Id.S*b[t]);
    end
    # find out on which days the extra order cost is incurred (i.e. weekend days)
    weekend_days = intersect(Id.T_sign, sort(union((6 - Id.L):7:Id.T[end], (7 - Id.L):7:Id.T[end], (8 - Id.L):7:(Id.T[end]))))
    # objective functions for different ordering policies
    if Id.policy == "NIS"
        @NLobjective(model, Min, # should the term including S̄m be omitted? Currently it is
            Id.P*sum(Id.O*q[t] + Id.H*v[t] + Id.E*e[t] + Id.G*f[t] for t in Id.T_sign)   # sum of basic ordering, holding, expiring and shortage costs
            + Id.P*sum(Id.Oe*q[t] for t in weekend_days)                             # sum of special ordering costs (weekends)
            - μ[xi] * Sm + rho2 / 2 * (S̄m - Sm)^2);                                 # Lagrangian multipliers
    elseif Id.policy == "1D" # use "NLobjective" for Juniper, "objective" for Gurobi
        @objective(model, Min,
            Id.P*sum(Id.O*q[t] + Id.H*v[t] + Id.E*e[t] + Id.G*f[t] for t in Id.T_sign)   # sum of basic ordering, holding, expiring and shortage costs
            + Id.P*sum(Id.Oe*b[t] for t in weekend_days)                             # sum of special ordering costs (weekends)
            + Id.P*sum(Id.Ot*b[t] for t in Id.T_sign)                                    # sum of transportation costs (incurred if at least one unit is ordered)
            + λ[xi] * (S̄tot - Stot) + rho1 / 2 * (S̄tot - Stot)^2);                  # Lagrangian multipliers
    else
        @NLobjective(model, Min,
            Id.P*sum(Id.O*q[t] + Id.H*v[t] + Id.E*e[t] + Id.G*f[t] for t in Id.T_sign)
            + Id.P*sum(Id.Oe*q[t] for t in weekend_days)
            + Id.P*sum(Id.Ot*b[t] for t in Id.T_sign) # PROBABLY DOESN'T WORK FOR 2D THIS WAY
            + λ[xi] * (S̄tot - Stot) + rho1 / 2 * (S̄tot - Stot)^2 + μ[xi] * (S̄m - Sm) + rho2 / 2 * (S̄m - Sm)^2);
    end

    # now all constraints that are the same for all policies
    @constraint(model, invbalance[t=Id.T, m=Id.M],      # inventory at the beginning = inventory at the end + used items
        is[t, m] == ie[t, m] + a[t, m]);                 # must hold for each age group m ∈ M

    ### Modified B into a decision variable, if solving SVP
    if Id.solving_SVP
        @constraint(model, startinventory[m=Id.M],
            is[1, m] == B[m]);
    ### if solving HOS, use preset initial inventory level
    else
        @constraint(model, startinventory[m=Id.M],     # inventory levels at t = 1
            is[1, m] == Id.B[m]);
    end
    @constraint(model, demand[t=Id.T],             # demand = used items + shortage
        sum(a[t, m] for m in Id.M) + f[t] == Id.D[xi, t]);
    @constraint(model, totalinv[t=Id.T],           # total inv in the end = everything but items with age 1 not used (will be outdated)
        v[t] == sum(ie[t, m] for m in Id.M[2:end]));
    @constraint(model, outdate[t=Id.T],            # outdate at t = old items (remaining life 1) not used
        e[t] == ie[t,1]);
    @constraint(model, invaging[t=Id.T[1:end-1], m=Id.M[1:end-1]],    # items of age a today = items of age a + 1 tomorrow
        ie[t, m+1] == is[t+1, m]);
    @constraint(model, ordertoinv[t=Id.T[1:end-Id.L]],                # items ordered in t = youngest items in stock in t + L
        is[t+Id.L, Id.M[end]] == q[t]);
    @constraint(model, service_level,     # service level must be met, CAUSES INFEASIBILITY IF USED IN UB COMPUTATION
     sum(a[t , m] for t in Id.T_sign, m in Id.M) >= Id.A*sum(Id.D[xi, t] for t in Id.T_sign));
    return model
end

@everywhere begin
    function parallel_lagr(xi::Int, Id::InstanceData,
            lambda::Array{Float64, 1}, mu::Array{Float64, 1})
        model_xi = create_aug_model(Id, lambda, mu, 0.0, 0.0, 0.0, 0.0, xi)
        status = optimize!(model_xi)
        return objective_value(model_xi)
    end
end

function solveLagrvalue(Id::InstanceData, λ::Array{Float64, 1}, μ::Array{Float64, 1},
        S̄tot::Float64, S̄m::Float64)
    # this function calculates Lagr dual value with given multiplier values
    # S̄m and S̄tot are not considered as Σλ_xi, Σμ_xi = 0 by def
    Lagr = pmap(parallel_lagr, Id.Xi, map(x -> Id, Id.Xi), map(x -> λ, Id.Xi), map(x -> μ, Id.Xi))
    return sum(Lagr)
end

# parallelUB solves the objective function for one scenario.
@everywhere begin
    function parallel_UB(xi::Int, Id::InstanceData, Stot::Float64, Sm::Float64)
        model_xi = create_aug_model(Id, zeros(length(Id.Xi)), zeros(length(Id.Xi)),
            0.0, 0.0, 0.0, 0.0, xi, true) # now ignore the service level constraint
        if Id.policy == "1D" || Id.policy == "2D"
            @constraint(model_xi, model_xi[:Stot] == Stot)
        end
        if Id.policy == "NIS" || Id.policy == "2D"
            @constraint(model_xi, model_xi[:Sm] == Sm)
        end
        status = optimize!(model_xi)

        a = value.(model_xi[:a])
        q = value.(model_xi[:q])
        e = value.(model_xi[:e])
        f = value.(model_xi[:f])
        return [objective_value(model_xi), a, q, e, f]
    end
end

# solveUB value gives the value of the intial obj fun with given order-up-to level Stot.
function solveUBvalue(Id::InstanceData, Stot::Float64, Sm::Float64)
    UB_and_srate = pmap(parallel_UB, Id.Xi, map(x -> Id, Id.Xi),
        map(x -> Stot, Id.Xi), map(x -> Sm, Id.Xi))
    UB = 0

    T = length(Id.T)
    M = length(Id.M)
    Xi = length(Id.Xi)

    a = zeros(T, M, Xi)
    q = zeros(T, Xi)
    e = zeros(T, Xi)
    f = zeros(T, Xi)
    for i in Id.Xi
        UB += UB_and_srate[i][1]
        a[:,:,i] = UB_and_srate[i][2]
        q[:,i] = UB_and_srate[i][3]
        e[:,i] = UB_and_srate[i][4]
        f[:,i] = UB_and_srate[i][5]
    end
    println("With order-up-to level of $(Stot), total expected cost is $(UB).")
    println("Total significant demand $(sum(Id.D[:,Id.T_sign])), total shortage $(sum(f[Id.T_sign,:])), total outdates $(sum(e[Id.T_sign,:]))")
    return UB, a, q, e, f

end

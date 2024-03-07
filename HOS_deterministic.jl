using DataFrames
using CSV
using Dates
using Gurobi        # requires licence

include("analysis_functions.jl")

### IMPORTANT ###
# Make sure your working directory is ".../Platelet-project", it is needed by the scenario loading
# use pwd() and cd() to check & change.

N = 8           # no of different demand instances per no of scenarios
                # i.e. the algorithm is run N times with newly created demand data
Xi = 60         # no of scenarios per set
M  = 5          # Platelets are usable for three days after they arrive in the hospital
T  = 28         # one scenario lasts for two weeks (starting on Monday) PLUS one week of 'warmup' and another of 'cooldown'
T_burnin = 7    # length of warmup and cooldown periods
servicelevel = 0.9
policy = "1D" # CHANGE THIS TO 2D and NIS IF NECESSARY, 1D used by default
scenariofiles = ["hosp1_17_19_demand.csv_4weeks.csv", "hosp2_17_19_demand.csv_4weeks.csv",
                 "hosp3_17_19_demand.csv_4weeks.csv", "hosp4_17_19_demand.csv_4weeks.csv",
                 "hosp5_17_19_demand.csv_4weeks.csv", "hosp6_17_19_demand.csv_4weeks.csv",
                 "med_17_19_demand.csv_4weeks.csv",   "small_17_19_demand.csv_4weeks.csv"]

# choose instance (this file solves only one per run)
i = 1
# upload right scenarios
scenarios = readscenarios(string(pwd(), "/Demand_data_17_19_4weeks/", scenariofiles[i]), Xi, T)

# create instane
Id = createinstance(Xi, T, T_burnin, M, scenarios, policy)
gurobienv = Gurobi.Env()

# create model
model = Model(with_optimizer(Gurobi.Optimizer, gurobienv, OutputFlag = 1))

# force all these to have only Int values
@variables model begin
    q[Id.T, Id.Xi] >= 0, Int                 # order quantity
    is[Id.T, Id.M, Id.Xi] >= 0, Int          # start inventory
    ie[Id.T, Id.M, Id.Xi] >= 0, Int          # end inventory
    a[Id.T, Id.M, Id.Xi] >= 0, Int           # no of units of age M used
    f[Id.T, Id.Xi] >= 0, Int                 # shortage (not dependent on the age)
    v[Id.T, Id.Xi] >= 0, Int                 # total inventory at end of t
    e[Id.T, Id.Xi] >= 0, Int                 # outdated
end

# add a few more variables & constraints depending on policy used
if Id.policy == "NIS"
    @variable(model, Sm >= 0, Int)           # amount to order in each round
    @constraint(model, NISorder[t=Id.T, xi=Id.Xi], q[t,xi] == Sm); #order quantity is always Sm
elseif Id.policy == "1D"
    @variables model begin
        Stot >= 0, Int                  # order-up-to level
        b[Id.T, Id.Xi], Bin                    # = 1 if there is shortage
    end
    #linearized constraints for order quantity
    @constraint(model, lin1[t=Id.T, xi=Id.Xi],    # amount to be ordered at least the amount of deficit
        q[t,xi] >= Stot - sum(is[t,m,xi] for m in Id.M[Id.L+1:end]));
    @constraint(model, lin2[t=Id.T, xi=Id.Xi],    # amount to be ordered not more than deficit (S̄ so that q >= 0 is possible)
        q[t,xi] <= Stot - sum(is[t,m,xi] for m in Id.M[Id.L+1:end]) + Id.S*(1-b[t,xi]));
    @constraint(model, lin3[t=Id.T, xi=Id.Xi],    # if no deficit, don't order anything
        q[t,xi] <= Id.S*b[t,xi]);
else    # dealing with 2D policy
    @variables model begin
        Stot >= 0, Int
        Sm >= 0, Int
        b[Id.T, Id.Xi], Bin                    # = 1 if deficit > Sm
    end
    # linearized constraints for order quantity
    @constraint(model, linear1[t=Id.T, xi=Id.Xi], # amount to be ordered at least the amount of deficit
        q[t,xi] >= Stot - sum(is[t,m,xi] for m in Id.M[Id.L+1:end]));
    @constraint(model, linear2[t=Id.T, xi=Id.Xi], # always order at least Sm items of fresh platelets
        q[t,xi] >= Sm);
    @constraint(model, linear3[t=Id.T], # amount to be ordered not more than deficit (S̄ so that q >= 0 is possible)
        q[t,xi] <= Stot - sum(is[t,m,xi] for m in Id.M[Id.L+1:end]) + Id.S*(1-b[t,xi]));
    @constraint(model, linear4[t=Id.T, xi=Id.Xi], # amount to be ordered = Sm if stock is already full (then b = 0)
        q[t,xi] <= Sm + Id.S*b[t,xi]);
end
# find out on which days the extra order cost is incurred (i.e. weekend days)
weekend_days = intersect(Id.T_sign, sort(union((6 - Id.L):7:Id.T[end], (7 - Id.L):7:Id.T[end], (8 - Id.L):7:(Id.T[end]))))
# objective functions for different ordering policies, use "NLobjective" for Juniper, "objective" for Gurobi
if Id.policy == "NIS"
    @objective(model, Min, # should the term including S̄m be omitted? Currently it is
        Id.P*sum(Id.O*q[t,xi] + Id.H*v[t,xi] + Id.E*e[t,xi] + Id.G*f[t,xi] for t in Id.T_sign, xi in Id.Xi)   # sum of basic ordering, holding, expiring and shortage costs
        + Id.P*sum(Id.Oe*q[t,xi] for t in weekend_days, xi in Id.Xi)                                 # sum of special ordering costs (weekends)
        + Id.P*sum(Id.Ot*b[t,xi] for t in Id.T_sign, xi in Id.Xi));                                   # sum of transportation costs (incurred if at least one unit is ordered)
elseif Id.policy == "1D"
    @objective(model, Min,
        Id.P*sum(Id.O*q[t,xi] + Id.H*v[t,xi] + Id.E*e[t,xi] + Id.G*f[t,xi] for t in Id.T_sign, xi in Id.Xi)   # sum of basic ordering, holding, expiring and shortage costs
        + Id.P*sum(Id.Oe*b[t,xi] for t in weekend_days, xi in Id.Xi)                                 # sum of special ordering costs (weekends)
        + Id.P*sum(Id.Ot*b[t,xi] for t in Id.T_sign, xi in Id.Xi));                                   # sum of transportation costs (incurred if at least one unit is ordered)
else
    @objective(model, Min,
        Id.P*sum(Id.O*q[t,xi] + Id.H*v[t,xi] + Id.E*e[t,xi] + Id.G*f[t,xi] for t in Id.T_sign, xi in Id.Xi)
        + Id.P*sum(Id.Oe*q[t,xi] for t in weekend_days, xi in Id.Xi)
        + Id.P*sum(Id.Ot*b[t,xi] for t in Id.T_sign, xi in Id.Xi)); # PROBABLY DOESN'T WORK FOR 2D THIS WAY
end

# now all constraints that are the same for all policies
@constraint(model, invbalance[t=Id.T, m=Id.M, xi=Id.Xi],      # inventory at the beginning = inventory at the end + used items
    is[t, m, xi] == ie[t, m, xi] + a[t, m, xi]);                 # must hold for each age group m ∈ M

### Modified B into a decision variable, if solving SVP
if Id.solving_SVP
    @constraint(model, startinventory[m=Id.M, xi=Id.Xi],
        is[1, m, xi] == B[m]);
### if solving HOS, use preset initial inventory level
else
    @constraint(model, startinventory[m=Id.M, xi=Id.Xi],     # inventory levels at t = 1
        is[1, m, xi] == Id.B[m]);
end
@constraint(model, demand[t=Id.T, xi=Id.Xi],             # demand = used items + shortage
    sum(a[t, m, xi] for m in Id.M) + f[t, xi] == Id.D[xi, t]);
@constraint(model, totalinv[t=Id.T, xi=Id.Xi],           # total inv in the end = everything but items with age 1 not used (will be outdated)
    v[t, xi] == sum(ie[t, m, xi] for m in Id.M[2:end]));
@constraint(model, outdate[t=Id.T, xi=Id.Xi],            # outdate at t = old items (remaining life 1) not used
    e[t, xi] == ie[t,1, xi]);
@constraint(model, invaging[t=Id.T[1:end-1], m=Id.M[1:end-1], xi=Id.Xi],    # items of age a today = items of age a + 1 tomorrow
    ie[t, m+1, xi] == is[t+1, m, xi]);
@constraint(model, ordertoinv[t=Id.T[1:end-Id.L], xi=Id.Xi],                # items ordered in t = youngest items in stock in t + L
    is[t+Id.L, Id.M[end], xi] == q[t, xi]);
@constraint(model, service_level[xi=Id.Xi],     # service level must be met, CAUSES INFEASIBILITY IF USED IN UB COMPUTATION
 sum(a[t, m, xi] for t in Id.T_sign, m in Id.M) >= Id.A*sum(Id.D[xi, t] for t in Id.T_sign));

println("Created the model, now solving")

# optimize model
status = optimize!(model)

# print interesting KPIs etc.
println("Objective value ", objective_value(model))
println("Optimal order-up-to level ", value(model[:Stot]))

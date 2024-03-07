using CSV

struct AnalysisData
    Xi::Int64
    M::Int64
    T::Int64
    T_burnin::Int64
    s_level::Float64
end

struct InstanceData
    Xi::UnitRange{Int64}            #Scenarios
    T::UnitRange{Int64}             #Time periods
    T_sign::UnitRange{Int64}        #Significant time period = T excluding burn-in periods
    M::UnitRange{Int64}             #Age groups of platelets
    B::Array{Int64, 2}              #Initial inventory for each age group
    G::Float64                      #Shortage cost
    H::Float64                      #Holding cost
    E::Float64                      #Expiry cost
    O::Float64                      #Order cost
    Oe::Float64                     #Extra order costs (weekends etc)
    Ot::Float64                     #Transportation cost
    L::Int64                        #Lead time
    A::Float64                      #required service level
    S::Int64                        #'Big M' for linearization of constraints
    P::Float64                      #Probability of a scenario
    D::Array{Float64, 2}            #Demand scenarios
    policy::String                  #NIS, 1D or 2D to determine policy used
    solving_SVP::Bool
end

include("hospital_fun.jl")

function createinstance(totalXi::Int64,
                        totalT::Int64,
                        T_burnin::Int64,
                        totalM::Int64,
                        scenarios::Array{Float64, 2},
                        policytype::String = "1D",
                        initialinv::Array{Int64, 2} = [10 10 10 10 10],
                        servicelevel::Float64 = 0.9)
    if 2 * T_burnin > totalT # if there are no 'significant' days in T
        throw("Invalid demand data, too long burn in periods")
    end
    Xi = 1:totalXi          # no of scenarios
    T = 1:totalT            # time periods (default = 14)
    T_sign = (T_burnin + 1):(totalT - T_burnin) # time periods that are included in the obj fun
    M = 1:totalM            # age groups (default = 3)
    B = initialinv          # inital inventory
    G = 822                 # shortage cost (= cost of emergency transportation etc?)
    H = 1                   # holding cost
    E = 411                 # expiry cost
    O = 411                 # order cost, weighted average of the to most popular products
    Oe = 14.13              # extra order cost (weekends)
    Ot = 91.34              # the basic transportation cost (per batch, not per item)
    L = 1                   # order lead time
    A = servicelevel        # required service level
    S = 60                  # 'Big M' (15)
    P = 1/length(Xi)        # probability of a scenario
    D = scenarios           # demand data

    if (size(D)[1] < totalXi) # if the data doesn't support given number of scenarios
       println("Using $(size(D)[1]) scenarios instead of $(totalXi). Effective shelf life is M = $(M).")
       Xi = 1:(size(D)[1])
    end
    policy = policytype     #NIS, 1D or 2D
    return InstanceData(Xi, T, T_sign, M, B, G, H, E, O, Oe, Ot, L, A, S, P, D, policy, false) # solving_SVP = false
end

function readscenarios(data_file::String, Xi::Int64, T::Int64)
    # this function assumes that each scenario is T days long
    data = DataFrame(CSV.File(data_file))

    total_entries = size(data,1)
    total_scenarios = div(total_entries, T) # number of scenarios

    scenarios = zeros(min(total_scenarios, Xi), T)
    for j in 1:min(total_scenarios, Xi)
        scenarios[j,:] = data[((j-1)*T + 1):(j*T), 3] # col index 3 corresponds to the column of the data file that contains the very demand number
    end

    return scenarios
end


# this function runs a single scenario and returns desired data
function runscenarios(policy::String, params::AnalysisData, scenarios::Array{Float64, 2})
    # create demand scenarios which are common for all algorithms
    ins = createinstance(params.Xi, params.T, params.T_burnin, params.M, scenarios, policy)

    idx_l = 1500

    Stot_all = zeros(idx_l, length(ins.Xi))    # space for all Stot_alls
    Sm_all = zeros(idx_l, length(ins.Xi))
    LB_all = zeros(idx_l)
    UB_all = zeros(idx_l)
    λ = zeros(idx_l, length(ins.Xi))
    μ = zeros(idx_l, length(ins.Xi))
    servicerate = zeros(idx_l)
    times = zeros(idx_l)
    # i::Int64    # no of iterations before stop

    Stot_all, Sm_all, LB_all, λ, μ, i, times = hospital_fun(ins)

    return Stot_all, Sm_all, LB_all, λ, μ, i, times, ins
end

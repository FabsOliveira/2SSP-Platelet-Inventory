using Random
using StatsBase
include("analysis_functions.jl")

# create instance for SVP model
function createSVPinstance(totalXi::Int64,
                          totalT::Int64,
                          T_burnin::Int64,
                          M_HOS::Int64,
                          M_SVP::Int64,
                          scenarios::Array{Float64, 2},
                          policytype::String = "1D",
                          # initialinv::Array{Int64, 2} = [70 70],
                          initialinv::Array{Int64, 2} = [150 150],
                          servicelevel::Float64 = 0.9)

    if 2 * T_burnin > totalT # if there are no 'significant' days in T
       throw("Invalid demand data, too long burn in periods")
    end
    Xi = 1:totalXi          # no of scenarios
    T = 1:totalT            # time periods (default = 14)
    T_sign = (T_burnin + 1):(totalT - T_burnin) # time periods that are included in the obj fun
    M = 1:M_SVP             # shelf life of the platelets at the blood bank
    B = initialinv          # inital inventory
    G = 10000                # shortage cost (= cost of emergency transportation etc?)
    H = 1                   # holding cost
    if M_HOS == 3
       O = 300                   # production cost
    else
       O = 400                 # order cost, pathogen
       # O = 330              # order cost, bacterial culture
    end
    E = O                   # expiry cost
    Oe = 0                  # extra order cost (weekends)
    Ot = 0                  # the basic transportation cost (per batch, not per item)
    L = 1                   # order lead time
    A = servicelevel        # required service level
    S = 500                 # 'Big M'
    P = 1/length(Xi)        # probability of a scenario
    D = scenarios           # demand data

    policy = policytype     #NIS, 1D or 2D
    return InstanceData(Xi, T, T_sign, M, B, G, H, E, O, Oe, Ot, L, A, S, P, D, policy, false) # solving_SVP = true
end

# create SVP level demand
function createSVPscenarios(totalXi::Int64,              # max no of scenarios included
                            totalT::Int64,               # scenario length
                            Xi_arr::Array{Int64, 1},     # vector of scenario numbers per hospital setting
                            Q::Array{Float64, 2},        # hospital order data
                            F::Array{Float64, 2})        # hospital emergency order data
   # create an amount totalXi of scenarios by summing hospital-wise Q, F
   rng = MersenneTwister(123) # fix the random number generator
   scenarios = zeros(totalXi, totalT)
   curr_Xi = 0  # keep track on scenarios used
   for i in 1:8 # do for each hospital setting
      if Xi_arr[i] >= totalXi
         Xi = shuffle(rng, 1:Xi_arr[i]) # shuffle the scenarios for each hospital to remove dependencies
      else
         Xi = vcat(shuffle(rng, 1:Xi_arr[i]), sample(rng, 1:Xi_arr[i], totalXi - Xi_arr[i]))
      end
      for xi in 1:totalXi
         scenarios[xi, :] += Q[:, curr_Xi + Xi[xi]] + F[:, curr_Xi + Xi[xi]] # sum selected Q and F (from the same scenario)
      end
      curr_Xi += Xi_arr[i] # use the scenarios of the next hospital in the next round
   end
   for xi in 1:totalXi, t in 1:totalT
      scenarios[xi, t] = round(scenarios[xi, t])
   end
   return scenarios # D matrix is of dimensions (totalXi, totalT)
end

@everywhere begin 
        using DataFrames
        using CSV
        using Dates
end

@everywhere include("analysis_functions.jl")

### IMPORTANT ###
# Make sure your working directory is ".../Platelet-project", it is needed by the scenario loading
# use pwd() and cd() to check & change.

N = 2           # no of different demand instances per no of scenarios
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

# Empty list for scenarios
scenarios = []

for i in 1:N
        push!(scenarios, readscenarios(string(pwd(), "/Demand_data_17_19_4weeks/", scenariofiles[i]), Xi, T))
end

Xi_arr = map(i -> size(scenarios[i])[1], 1:N) # array that contains number of scenarios per setting
Xi_sum = sum(Xi_arr) # sum of all scenarios

params = AnalysisData(Xi, M, T, T_burnin, servicelevel)

# REWRITE THE ANALYSIS SCRIPT

n_iter = 1500       # number of iterations done (at most) HARD CODING, FIX!

Stot_all = zeros(n_iter, Xi, N) # dimensions: iterations x scenarios x hospital no.
Sm_all = zeros(n_iter, Xi, N)
LB_all = zeros(n_iter, N)
λ = zeros(n_iter, Xi, N)
μ = zeros(n_iter, Xi, N)
n_iter_realised = zeros(N)      # how many iterations were made befor algorithm stopped
times = zeros(n_iter, N)        # time needed by the algorithm for each iteration
ins = Array{InstanceData, 1}(undef, N)

# arrays for storing the final UB results
a = zeros(T, M, Xi_sum) # no of platelets assigned (for each age class)
q = zeros(T, Xi_sum)    # order amount
e = zeros(T, Xi_sum)    # expiry
f = zeros(T, Xi_sum)    # shortage

# Create a data frame to store info about the current iterations
res_df = DataFrame(settingno = Int64[],
                   no_of_scenarios = Int64[],
                   avg_final_Stot = Float64[],
                   Stot_for_UB = Int64[],
                   objval = Float64[])
res_header =  ["Hospital no.", "#Scenarios", "Avg. final S_tot", "S_tot for UB", "Obj val"]

print("\n\nRunning scenarios for $(N) hospitals, using Augmented Lagrangian, Nesterov subgradient algorithm \n–---–\n")

Xi_curr = 0 # count how many scenarios have been gone through
for i in 1:N
        t_ini = time()
        Xi_range = 1:Xi_arr[i]    # get the no of scenarios in this round
        Xi_int = Xi_curr .+ Xi_range

        println("### Hospital $(i)")
        res = runscenarios(policy, params, scenarios[i])

        # store iteration results
        Stot_all[:,Xi_range,i] = res[1]
        Sm_all[:,Xi_range,i]   = res[2]
        LB_all[:,i]            = res[3]
        λ[:,Xi_range,i]        = res[4]
        μ[:,Xi_range,i]        = res[5]
        n_iter_realised[i]     = res[6]
        times[:,i]             = res[7]
        ins[i]                 = res[8]

        # store UB results
        UB, a[:, :, Xi_int], q[:, Xi_int], e[:, Xi_int], f[:, Xi_int] =
                solveUBvalue(ins[i], ceil(maximum(Stot_all[Int(n_iter_realised[i]),ins[i].Xi,i])), 0.0)
        global Xi_curr += Xi_arr[i]

        # the mean of the S_tots found in the last iteration round
        mean_Stot = mean(Stot_all[res[6],Xi_range,i])
        push!(res_df, [i, length(ins[i].Xi), mean_Stot, ceil(maximum(Stot_all[Int(n_iter_realised[i]),ins[i].Xi,i])), UB])

        println("---––\n")
        t_end = time() - t_ini
        println("total time ($(nprocs()) workers): $t_end")
end

# Create a folder for today's results
new_folder = "Results ($(M) days M) $(Dates.today())"
# if a folder with similar name already exists, rename
if ispath(string(pwd(), "/", new_folder))
        i = 1
        while ispath(string(pwd(), "/", new_folder, "_$(i)"))
                i += 1
        end
        new_folder = string(new_folder, "_$(i)")
end
mkdir(new_folder)
curr_dir = pwd() # save current directory

# a is a 3D array, so it must be reduced to 2D to convert it to data frame
a_df = DataFrame([a[t,:,xi] for t in 1:T, xi in 1:Xi_sum])

header = [] # make header for the CSV files
for i in 1:length(Xi_arr)
        append!(header, map(x -> string("H$(i)S", string(x)), 1:Xi_arr[i]))
end

cd(new_folder) # go to new folder and save results
CSV.write("KPIs_HOS.csv", res_df,       header = res_header, delim = '\t')
CSV.write("A_HOS.csv",    a_df,         header = header,      delim = '\t')
CSV.write("Q_HOS.csv",    DataFrame(q), header = header,      delim = '\t')
CSV.write("E_HOS.csv",    DataFrame(e), header = header,      delim = '\t')
CSV.write("F_HOS.csv",    DataFrame(f), header = header,      delim = '\t')
cd(curr_dir) # return to curr_dir

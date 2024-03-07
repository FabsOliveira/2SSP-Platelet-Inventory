using DataFrames
using CSV
using Dates

include("SVP_functions.jl")


### Note! Make sure your pwd() is ".../Platet-project"
### Note! Set 'res_folder ' as the name of the folder you are using to extract the HOS results by hand!
Xi = 60
T = 28
T_burnin = 7
M_HOS = 5
M_SVP = 2

# create demand scenarios for the central blood bank:
# use both order data q and shortage data f. No of scenarios per HOS setting is also needed.
curr_dir = pwd()
res_folder = "Results 2020-12-07-all/Results (5 days M) 2020-12-07/"
cd(res_folder)
Q = Array(CSV.read("Q_HOS.csv"))
F = Array(CSV.read("F_HOS.csv"))
Xi_arr = CSV.read("KPIs_HOS.csv")[:,2]
cd(curr_dir)

# create scenarios and the instance
SVP_scenarios = createSVPscenarios(Xi, T, Xi_arr, Q, F)
SVP_ins = createSVPinstance(Xi, T, T_burnin, M_HOS, M_SVP, SVP_scenarios)


# create new instance data frame where cost params are set for SVP level
SVP_res_df = DataFrame(no_of_scenarios = Int64[],
                        avg_final_Stot = Float64[],
                        stot_for_UB = Int64[],
                        objval = Float64[])
SVP_res_header =  ["#Scenarios", "Avg. final S_tot", "S_tot for UB", "Obj val"]

# arrays for storing the final UB results
SVP_a = zeros(T, M_SVP, Xi) # total no of platelets delivered (for each age class)
SVP_q = zeros(T, Xi)    # order amount
SVP_e = zeros(T, Xi)    # expiry
SVP_f = zeros(T, Xi)    # shortage

# run the algorithm, then compute UB value just as in analysis.jl.
println("\n### SVP setting")
Stot_all, Sm_all, LB_all, λ, μ, n_iter, t = hospital_fun(SVP_ins)
println("Avg. Stot of the last iteration: ", mean(Stot_all[n_iter, :]))
SVP_UB, SVP_a, SVP_q, SVP_e, SVP_f = solveUBvalue(SVP_ins, ceil(maximum(Stot_all[Int(n_iter), SVP_ins.Xi])), 0.0)
println("\nSignificant period:\nTotal demand $(sum(SVP_scenarios[:,SVP_ins.T_sign]))")
println("Total shortage $(sum(SVP_f[SVP_ins.T_sign, :]))")
println("Total outdate $(sum(SVP_e[SVP_ins.T_sign, :]))")
push!(SVP_res_df, [Xi, mean(Stot_all[Int(n_iter), SVP_ins.Xi]), ceil(maximum(Stot_all[Int(n_iter), SVP_ins.Xi])), SVP_UB])


# Create a folder for today's results
new_folder = "SVP results ($(M_HOS) days M) $(Dates.today())"
# if a folder with similar name already exists, rename
if ispath(string(pwd(), "/", new_folder))
        i = 1
        while ispath(string(pwd(), "/", new_folder, "_$(i)"))
                i += 1
        end
        new_folder = string(new_folder, "_$(i)")
end
mkdir(new_folder)

# a is a 3D array, so it must be reduced to 2D to convert it to data frame
SVP_a_df = DataFrame([SVP_a[t,:,xi] for t in 1:T, xi in 1:Xi])
header = map(xi -> string(xi), 1:Xi)

cd(new_folder) # go to new folder and save results
CSV.write("KPIs_SVP.csv", SVP_res_df,       header = SVP_res_header, delim = '\t')
CSV.write("A_SVP.csv",    SVP_a_df,         header = header,         delim = '\t')
CSV.write("Q_SVP.csv",    DataFrame(SVP_q), header = header,         delim = '\t')
CSV.write("E_SVP.csv",    DataFrame(SVP_e), header = header,         delim = '\t')
CSV.write("F_SVP.csv",    DataFrame(SVP_f), header = header,         delim = '\t')
cd(curr_dir) # return to curr_dir

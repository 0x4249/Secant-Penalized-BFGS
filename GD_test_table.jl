using CUTEst
using Distributions
using LinearAlgebra
using Printf
using Random
include("noiseFunctions.jl")
include("objectiveFunctions.jl")

# Initialize objective function
#nlp = CUTEstModel("SBRYBND", "-param", "N=10")
nlp = CUTEstModel("ROSENBR")
d = length(nlp.meta.x0)
objFunTrue = nlp_objective(nlp)

# Set minimum objective value according to SIF file for CUTEst problem above
obj_fun_min_val = 0.0

# Set gradient noise level
#epsilon_g = (1e-4)*norm(objFunTrue.grad_eval(nlp.meta.x0))
epsilon_g = 1e2

# Set function noise level
#epsilon_f = (1e-4)*abs(objFunTrue.func_eval(nlp.meta.x0))
epsilon_f = 1e0

# Set noise type
#objFun = nlp_objective_additive_noise(nlp,zero_noise(),ball_uniform_random_noise(d, epsilon_g))
objFun = nlp_objective_additive_noise(nlp,symmetric_uniform_random_noise(epsilon_f),ball_uniform_random_noise(d, epsilon_g))

# Aggregate parameters
total_runs = 30
max_obj_fun_evals = 2000
best_true_obj_vals = zeros(total_runs)
line_search_fail_counts = zeros(total_runs)
iter_counts = zeros(total_runs)

# Repeated algorithm runs
for r in 1:total_runs
	global max_obj_fun_evals, iter_counts, best_true_obj_vals
	
	# Optimization setup
	x_0 = nlp.meta.x0
	
	# Backtracking line search parameters
	alpha_0 = 1
	c_1 = 1e-4
	tau = 0.5
	noise_tol = epsilon_f
	max_backtracks = 45
	
	# Termination parameters
	termination_eps = 0
	
	# Allocate storage
	line_search_fail_count = 0
	obj_fun_eval_count = 0
	best_true_obj_val = Inf
	iter_count = 0
	
	# Gradient Descent Loop
	x = x_0
	f_old = Inf
	f_old_true = Inf
	g_old = zeros(d)
	
	while obj_fun_eval_count < max_obj_fun_evals
		iter_count += 1
		alpha = alpha_0
		
		# Initial noisy function and gradient evalutions
		if obj_fun_eval_count == 0
			f_old = objFun.func_eval(x)
			f_old_true = objFunTrue.func_eval(x)
			g_old = objFun.grad_eval(x)
			obj_fun_eval_count += 1
		end
		
		x_old = x
		p = -g_old
		x = x_old + alpha*p
		
		f_new = objFun.func_eval(x)
		f_new_true = objFunTrue.func_eval(x)
		g_new = objFun.grad_eval(x)
		obj_fun_eval_count += 1
		
		# Backtracking line search
		line_search_fail = 0
		while f_new > f_old + c_1*alpha*dot(g_old,p) + 2*noise_tol && obj_fun_eval_count < max_obj_fun_evals
			line_search_fail +=1
		  	if line_search_fail > max_backtracks
		  		alpha = 0
		  	else
				alpha = tau*alpha
		  	end
		  	x = x_old + alpha*p		 
		  	f_new = objFun.func_eval(x)
		  	f_new_true = objFunTrue.func_eval(x)
		  	g_new = objFun.grad_eval(x)
		  	obj_fun_eval_count += 1	      
		end
		line_search_fail_count += line_search_fail
		
		@show(line_search_fail)
		#@show(i,x,f_old,f_new)
		@show(obj_fun_eval_count,f_old,f_new)
		@show norm(objFunTrue.grad_eval(x))
		
		if norm(g_new) <= termination_eps
			@printf("Terminated within tolerance after %d iterations\n", i)
			break
		end
		
		best_true_obj_val = min(f_old_true,f_new_true,best_true_obj_val)
		f_old = f_new
		f_old_true = f_new_true
		g_old = g_new
		@show r
	end
	
	@show(line_search_fail_count)
	
	best_true_obj_vals[r] = best_true_obj_val
	line_search_fail_counts[r] = line_search_fail_count
	iter_counts[r] = iter_count
end

# Print table statistics
@printf("====Table Values====\n")
@printf("GD Mean Log10 Optimality Gap: %1.1E\n", mean(log.(10,best_true_obj_vals.-obj_fun_min_val)))
@printf("GD Median Log10 Optimality Gap: %1.1E\n", median(log.(10,best_true_obj_vals.-obj_fun_min_val)))
@printf("GD Minimum Log10 Optimality Gap: %1.1E\n", minimum(log.(10,best_true_obj_vals.-obj_fun_min_val)))
@printf("GD Maximum Log10 Optimality Gap: %1.1E\n", maximum(log.(10,best_true_obj_vals.-obj_fun_min_val)))
@printf("GD Log10 Optimality Gap Variance: %1.1E\n", var(log.(10,best_true_obj_vals.-obj_fun_min_val)))
@printf("GD Mean Number of Iterations: %d\n", mean(iter_counts))


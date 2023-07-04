using Distributions
using LinearAlgebra
using Printf
using PyPlot
using Random
include("noiseFunctions.jl")
include("objectiveFunctions.jl")

# Initialize objective function
eigenvalues = [1e-2,1,1e2,1e4]
d = length(eigenvalues)
Q = diagm(eigenvalues)
epsilon_g = 1
objFun = quadratic_additive_noise(zero_noise(),ball_uniform_random_noise(d, epsilon_g),A=Q)
objFunTrue = quadratic(A=Q) 

# Aggregate parameters
total_runs = 30
maxIter = 100
curv_fail_counts = zeros(total_runs)
line_search_fail_counts = zeros(total_runs)
all_f_vals = zeros(maxIter,total_runs)
all_x_vals = zeros(maxIter,d,total_runs)
all_scaled_hess_cond_vals = zeros(maxIter,total_runs)
all_line_search_fails = zeros(maxIter,total_runs)
all_alphas = zeros(maxIter,total_runs)

# Plotting stuff
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["text.usetex"] = true
rcParams["figure.autolayout"] = true
color_map = get_cmap("twilight_shifted", total_runs)

# Repeated algorithm runs
for r in 1:total_runs
	global all_f_vals, all_x_vals, all_line_search_fails, all_alphas, maxIter
	
	# Optimization setup
	x_0 = 1e5.*[1;1;1;1]
	H_0 = I
	
	# Backtracking line search parameters
	alpha_0 = 1
	c_1 = 1e-4
	tau = 0.5
	noise_tol = 0
	max_backtracks = 75
	
	# Termination parameters
	termination_eps = 1e-5
	
	# Allocate storage
	curv_fail_count = 0
	line_search_fail_count = 0
	f_vals = zeros(maxIter)
	x_vals = zeros(maxIter,d)
	scaled_hess_cond_vals = zeros(maxIter)
	line_search_fails = zeros(maxIter)
	alphas = zeros(maxIter)
	
	# BFGS Loop
	x = x_0
	H = H_0
	f_old = 0 
	g_old = [0; 0]
	
	for i in 1:maxIter
		alpha = alpha_0
		
		# Initial noisy function and gradient evalutions
		if i == 1
			f_old = objFun.func_eval(x)
			g_old = objFun.grad_eval(x)
		end
		
		x_old = x
		p = -H*g_old
		x = x_old + alpha*p
		
		f_new = objFun.func_eval(x)
		g_new = objFun.grad_eval(x)
		
		# Backtracking line search
		line_search_fail = 0
		while f_new > f_old + c_1*alpha*dot(g_old,p) + 2*noise_tol
			line_search_fail +=1
		  	if line_search_fail > max_backtracks
		  		alpha = 0
		  	else
			  	alpha = tau*alpha
		  	end
			x = x_old + alpha*p		 
		  	f_new = objFun.func_eval(x)
	      	  	g_new = objFun.grad_eval(x)
		end
		line_search_fail_count += line_search_fail
		line_search_fails[i] = line_search_fail
		alphas[i] = alpha
		
		# Calculate secant condition parameters
		s_k = x - x_old
		y_k = g_new - g_old
		y_k_dot_s_k = dot(s_k,y_k)
		
		# Check curvature condition
		if y_k_dot_s_k > 0
			rho_k = 1/y_k_dot_s_k
			@show rho_k
			
			H = (I - rho_k*s_k*y_k')*H*(I - rho_k*y_k*s_k') + rho_k*s_k*s_k'
		else 
		    	@printf("Curvature condition failed!\n")
		    	curv_fail_count += 1
		end
		
		@show(line_search_fail)
		@show(i,x,f_old,f_new)
		
		if norm(g_new) <= termination_eps
			@printf("Terminated within tolerance after %d iterations\n", i)
			break
		end
		
		f_vals[i] = f_new
		x_vals[i,:] = x
		scaled_hess_cond_vals[i] = cond(H*objFun.hess_eval(x))
		f_old = f_new
		g_old = g_new
		@show r
	end
	
	@show(curv_fail_count)
	@show(line_search_fail_count)
	@show(norm(H))
	@show(objFun.hess_eval(x))
	
	curv_fail_counts[r] = curv_fail_count
	line_search_fail_counts[r] = line_search_fail_count
	all_f_vals[:,r] = f_vals
	all_x_vals[:,:,r] = x_vals
	all_scaled_hess_cond_vals[:,r] = scaled_hess_cond_vals
	all_line_search_fails[:,r] = line_search_fails
	all_alphas[:,r] = alphas
	
end

# Optimality gap figure
pltIter = maxIter
figure(4)
axes_opt_gap = plt.gca()
axes_opt_gap.set_xlim([-1,pltIter])
#axes.set_yscale("log")
axes_opt_gap.set_ylim([-10,15])
axes_opt_gap.grid(true)
axes_opt_gap.tick_params(axis="x", labelsize=20)
axes_opt_gap.tick_params(axis="y", labelsize=20)
xlabel("Iteration k", fontsize=22)
ylabel(L"$\log_{10}(\phi_k - \phi^{\star})$", fontsize=22)
title("BFGS Optimality Gap", fontsize=22)
for r in 1:total_runs
	axes_opt_gap.plot(1:pltIter,log.(10,all_f_vals[:,r]), alpha=0.8, color=color_map(r/total_runs), label="BFGS")
end

# Gradient norm figure
figure(5) 
axes_grad_norm = plt.gca()
axes_grad_norm.set_xlim([-1,pltIter])
#axes.set_yscale("log")
axes_grad_norm.set_ylim([-2.5,10])
axes_grad_norm.grid(true)
axes_grad_norm.tick_params(axis="x", labelsize=20)
axes_grad_norm.tick_params(axis="y", labelsize=20)
xlabel("Iteration k", fontsize=22)
ylabel(L"$\log_{10}(\left\| \nabla \phi_k \right\|_2)$", fontsize=22)
title("BFGS Gradient Norm", fontsize=22)
all_grad_norms = zeros(maxIter,total_runs)
for r in 1:total_runs
	for j in 1:maxIter
		all_grad_norms[j,r] = norm(objFunTrue.grad_eval(all_x_vals[j,:,r]))
	end
	axes_grad_norm.plot(1:pltIter,log.(10,all_grad_norms[:,r]), alpha=0.8, color=color_map(r/total_runs), label="BFGS")
end

# Scaled Hessian figure
figure(6)
axes_scaled_Hessian = plt.gca()
axes_scaled_Hessian.set_xlim([-1,pltIter])
axes_scaled_Hessian.set_ylim([-1,25])
axes_scaled_Hessian.grid(true)
axes_scaled_Hessian.tick_params(axis="x", labelsize=20)
axes_scaled_Hessian.tick_params(axis="y", labelsize=20)
xlabel("Iteration k", fontsize=22)
ylabel(L"$\log_{10} \bigg (\kappa \big ( H_k \nabla^2 \phi_k \big ) \bigg )$", fontsize=22)
title("BFGS Condition Number", fontsize=22)
for r in 1:total_runs
	axes_scaled_Hessian.plot(1:pltIter,log.(10,all_scaled_hess_cond_vals[:,r]), alpha=0.8, color=color_map(r/total_runs), label="BFGS")
end


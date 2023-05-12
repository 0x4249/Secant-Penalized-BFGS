using LinearAlgebra
using NLPModels

# Define an "ObjFun" type for evaluating an objective function and its derivatives
mutable struct ObjFun
	func_eval # Function that evaluates the objective function
	grad_eval # Function that evaluates the gradient of the objective function
	hess_eval # Function that evaluates the Hessian of the objective function
end

# Function to evaluate quadratic function with matrix A
function quadratic(;A=[5 1; 1 3]) 
	func_eval(x) = (1/2)*x'*A*x		
	grad_eval(x) = (1/2)*(A+A')*x
	hess_eval(x) = (1/2)*(A+A')
	return ObjFun(func_eval,grad_eval,hess_eval)
end

# Function to evaluate quadratic function with matrix A
# with noise added component-wise to the function and gradient evaluations
function quadratic_additive_noise(func_noise,grad_noise;A=[5 1; 1 3]) 
	func_eval(x) = (1/2)*x'*A*x .+ func_noise(x)
	grad_eval(x) = (1/2)*(A+A')*x .+ grad_noise(x)
	hess_eval(x) = (1/2)*(A+A')
	return ObjFun(func_eval,grad_eval,hess_eval)
end

# Function to evaluate an NLPModel from the NLPModels.jl API
function nlp_objective(nlp; obj_weight=1.0)
	func_eval(x) = obj(nlp, x)
	grad_eval(x) = grad(nlp, x)
	L(x) = hess(nlp, x; obj_weight=obj_weight)
	function hess_eval(x)
		B = L(x)
		return Array(B) + Array(B') - Diagonal(diag(B))
	end
	return ObjFun(func_eval,grad_eval,hess_eval)
end 

# Function to evaluate an NLPModel from the NLPModels.jl API
# with noise added component-wise to the function and gradient evaluations
function nlp_objective_additive_noise(nlp,func_noise,grad_noise; obj_weight=1.0)
	func_eval(x) = obj(nlp, x) .+ func_noise(x)
	grad_eval(x) = grad(nlp, x) .+ grad_noise(x)
	L(x) = hess(nlp, x; obj_weight=obj_weight)
	function hess_eval(x)
		B = L(x)
		return Array(B) + Array(B') - Diagonal(diag(B))
	end
	return ObjFun(func_eval,grad_eval,hess_eval)
end 


using Distributions
using LinearAlgebra
using Random

# Function returning a function representing scalar zero noise
function zero_noise()
	noise(x) = return 0
	return noise
end

# Function returning a function representing scalar unit noise
function unit_noise()
	noise(x) = return 1
	return noise
end

# Function returning a function representing scalar symmetric uniform random noise
function symmetric_uniform_random_noise(epsilon=1e-2)
	noise(x) = rand(Uniform(-epsilon,epsilon))
	return noise
end

# Function returning a function representing scalar Gaussian random noise
function gaussian_random_noise(mu=0,sigma=1)
	noise(x) = mu + sigma*randn()
	return noise
end

# Function returning a function representing noise uniformly distributed inside 
# the d-dimensional ball with radius r
function ball_uniform_random_noise(d=2, r=1)
	function ball_uniform_random_sample(d,r)
		y = randn(d)
		Y = y./norm(y)
		U = rand().^(1/d)
		return r*U*Y
	end
	noise(x) = ball_uniform_random_sample(d,r)
	return noise
end


+++
title = 'Bayesian Time Series Analysis with Julia and Turing.jl'
date = 2024-03-02T16:57:07+08:00
draft = false
summary = "This tutorial covers the fundamentals of Bayesian approaches to time series, model construction, and practical implementation, using real-world data for hands-on learning."
tags = ["Bayesian", "Time Series", "Regression", "Turing", "Julia"]
+++
---

## Introduction

In this tutorial, an AR(p) (Autoregressive model of order _p_) is employed to analyze the trneds of a time series and forecast the behavior of the signal.

Auto-regressive models are based on the assumption the behavior of a time series or signal depends on past values. The order of the AR model tells "how far back" the past values will affect the current value.

#### Credits

This exercise is mostly following [this tutorial](https://youtu.be/vfTYCm_Fr8I?si=D3Grgk82tV_Qzdxw).

### Definition

The _AR(p)_ model is defined as:

$$
X_t = \sum_{i=1}^{p} \phi_i X_{t-i} + \varepsilon_t
$$

where $\varepsilon \sim \mathcal{N}(0,\sigma^2)$ is the model uncertainty represented as white Gaussian noise, i.e. it follows a normal distribution of mean $\mu=0$ and standard deviation $\sigma$.

It follows that an _AR(2)_ model is defined as:

$$
X_t = \phi_1 X_{t-1} + \phi_2 X_{t-2} + \varepsilon_t
$$

Naturally, we want to find the parameters $\theta=\{\phi_1, \phi_2,\sigma\}$. Since these are unobserved quantities of interest, we need to use an inference method to reveal these parameters. We will use Bayesian inference to achieve this goal.

## Data Exploration

For this example, I will generate artificial data. This will be done by first defining some values for the parameters $\theta$ and then we will generate random data using those parameters by initializing the $X_1, X_2$ values, and then applying the AR(2) equation to generate the subsequent values.

First, we import the relevant packages.


```julia
using StatsPlots, Turing, LaTeXStrings, Random, DataFrames
Random.seed!(42)
```




    TaskLocalRNG()



Now we create some artificial data. The steps involved in this are as follows:
1. Define some values for the parameters $\theta$
2. Set the number of timesteps _t_
3. Initialize an empty vector of size $\mathbb{R}^{t+p}$
4. Initialize the first two $X$ values with randomly generated numbers using `rand`
5. Populate the vector by using the equation for $X_t$


```julia
# define true values for θ
true_phi_1 = -0.4
true_phi_2 = 0.3
true_sigma = 0.12

# define the time steps
time = 100
	
# create an empty X vector
X = Vector{Float64}(undef, time+2)

# initialize the X vector with two random values at time steps 1 and 2
# to do this, use a random normally distributed number with mean zero and standard deviation σ, i.e., ε~N(0, σ)
X[1] = rand(Normal(0, true_sigma))
X[2] = rand(Normal(0, true_sigma))

# populate vector X
for t in 3:(time+2)
	X[t] = true_phi_1*X[t-1] +
	true_phi_2*X[t-2] +
	rand(Normal(0, true_sigma))
end	
```

### Visualize the (Artificial) Data


```julia
p_data = plot(X[3:end],
    legend = false,
    linewidth = 2,
    # xlims = (0, 60),
    # ylims = (-0.6, 0.6),
    title = "Bayesian Autoregressive AR(2) Model",
    xlabel = L"t",
    ylabel = L"X_t",
    widen = true
)
```


![svg](/images/20240222_Bayesian_Time_Series_Analysis/output_5_0.svg)

## Modeling 

The next step is to construct our probabilistic model. Again, the goal here is to infer the values of the model parameters $\theta$. Once we have inferred these parameters, we can make probabilistic predictions on the future behavior of the signal $X$.

### Bayesian model

Since we are using a Bayesian approach, our goal, in Bayesian terms, is to find the _posterior distribution_ of the parameters $\theta$, given a prior distribution, or prior knowledge, of the parameters before making any observations, i.e., seeing any data, and also a likelihood function, which reflects what kind of distribution (we assume) that the data is sourced from. Another way of understanding the likelihood function is the probability of making a set of observations $X$ given the parameters $\theta$. 

This relationship is established by Bayes' Theorem:

$$
P(\theta | X) \propto P(X | \theta)P(\theta)
$$

In summary, constructing the Bayesian model in this case comprises a selection of prior distributions for our unknown parameters $\theta$ and a likelihood function. We will do this using the `Turing.jl` package.

The model therefore will consist of the prior distributions:

$$
\begin{align*}
\phi_1 & \sim \mathcal{N}(0, 1) \\
\phi_2 & \sim \mathcal{N}(0, 1) \\
\sigma & \sim \text{Exp}(1)
\end{align*}
$$

And the likelihood:

$$
X_t \sim \mathcal{N}(\mu_t, \sigma)
$$

where $\mu_t = \sum_{i=1}^{p} \phi_i X_{t-i}$ is the mean function of the distribution that governs X_t.

#### A comment on the choice of priors

For autoregressive parameters, using a normal distribution is a common choice. This is because the normal distribution is convenient and allows for a range of plausible values.

For the prior on the model uncertainty, the exponential distribution is sometimes used for non-negative parameters and has a similar role to the inverse gamma.

Furthermore, the inverse gamma distribution is often chosen as a prior for the standard deviation because it is conjugate to the normal likelihood. This means that the posterior distribution will have a known form, making computations more tractable.

### Bayesian model using `Turing.jl`

Now we proceed to set up the model using the `Turing.jl` package.



```julia
@model function ar(X, time)    # pass the data X and the time vector

		# priors
		
		phi_1 ~ Normal(0, 1)
		phi_2 ~ Normal(0, 1)
		sigma ~ Exponential(1)

		# likelihood

		# initialize with random initial values
		X[1] ~ Normal(0, sigma)
		X[2] ~ Normal(0, sigma)

		# populate with samples
		for i in 3:(time+2)
			mu = phi_1*X[i-1] + phi_2*X[i-2]
			X[i] ~ Normal(mu, sigma)
		end
	end
```




    ar (generic function with 2 methods)




```julia
model = ar(X, time)
sampler = NUTS()
samples = 1_000

chain = sample(model, sampler, samples)
```

    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39mFound initial step size
    [36m[1m└ [22m[39m  ϵ = 0.4
    [32mSampling: 100%|█████████████████████████████████████████| Time: 0:00:01[39m
    




    Chains MCMC chain (1000×15×1 Array{Float64, 3}):
    
    Iterations        = 501:1:1500
    Number of chains  = 1
    Samples per chain = 1000
    Wall duration     = 11.59 seconds
    Compute duration  = 11.59 seconds
    parameters        = phi_1, phi_2, sigma
    internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size
    
    Summary Statistics
     [1m parameters [0m [1m    mean [0m [1m     std [0m [1m    mcse [0m [1m ess_bulk [0m [1m ess_tail [0m [1m    rhat [0m [1m e[0m ⋯
     [90m     Symbol [0m [90m Float64 [0m [90m Float64 [0m [90m Float64 [0m [90m  Float64 [0m [90m  Float64 [0m [90m Float64 [0m [90m  [0m ⋯
    
           phi_1   -0.3830    0.1047    0.0036   836.6151   762.4445    0.9996     ⋯
           phi_2    0.1587    0.1012    0.0035   838.3014   749.6718    1.0002     ⋯
           sigma    0.1083    0.0079    0.0003   755.4034   743.3822    1.0014     ⋯
    [36m                                                                1 column omitted[0m
    
    Quantiles
     [1m parameters [0m [1m    2.5% [0m [1m   25.0% [0m [1m   50.0% [0m [1m   75.0% [0m [1m   97.5% [0m
     [90m     Symbol [0m [90m Float64 [0m [90m Float64 [0m [90m Float64 [0m [90m Float64 [0m [90m Float64 [0m
    
           phi_1   -0.5733   -0.4562   -0.3858   -0.3141   -0.1771
           phi_2   -0.0339    0.0913    0.1562    0.2256    0.3549
           sigma    0.0943    0.1030    0.1079    0.1130    0.1257
    



### Visualize and Summarize the Results

Next we can access the MCMC Diagnostics and generate a summary of the results.


```julia
plot(chain)
```

![svg](/images/20240222_Bayesian_Time_Series_Analysis/output_10_0.svg)

```julia
DataFrame(summarystats(chain))
```




<div><div style = "float: left;"><span>3×8 DataFrame</span></div><div style = "clear: both;"></div></div><div class = "data-frame" style = "overflow-x: scroll;"><table class = "data-frame" style = "margin-bottom: 6px;"><thead><tr class = "header"><th class = "rowNumber" style = "font-weight: bold; text-align: right;">Row</th><th style = "text-align: left;">parameters</th><th style = "text-align: left;">mean</th><th style = "text-align: left;">std</th><th style = "text-align: left;">mcse</th><th style = "text-align: left;">ess_bulk</th><th style = "text-align: left;">ess_tail</th><th style = "text-align: left;">rhat</th><th style = "text-align: left;">ess_per_sec</th></tr><tr class = "subheader headerLastRow"><th class = "rowNumber" style = "font-weight: bold; text-align: right;"></th><th title = "Symbol" style = "text-align: left;">Symbol</th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Float64" style = "text-align: left;">Float64</th></tr></thead><tbody><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">1</td><td style = "text-align: left;">phi_1</td><td style = "text-align: right;">-0.383019</td><td style = "text-align: right;">0.104695</td><td style = "text-align: right;">0.00361324</td><td style = "text-align: right;">836.615</td><td style = "text-align: right;">762.444</td><td style = "text-align: right;">0.999585</td><td style = "text-align: right;">72.1655</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">2</td><td style = "text-align: left;">phi_2</td><td style = "text-align: right;">0.158661</td><td style = "text-align: right;">0.101196</td><td style = "text-align: right;">0.00351463</td><td style = "text-align: right;">838.301</td><td style = "text-align: right;">749.672</td><td style = "text-align: right;">1.00021</td><td style = "text-align: right;">72.311</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">3</td><td style = "text-align: left;">sigma</td><td style = "text-align: right;">0.108342</td><td style = "text-align: right;">0.00788622</td><td style = "text-align: right;">0.000291067</td><td style = "text-align: right;">755.403</td><td style = "text-align: right;">743.382</td><td style = "text-align: right;">1.00145</td><td style = "text-align: right;">65.1603</td></tr></tbody></table></div>



## Predictions

### Making Predictions

To make predictions, the following steps are taken:

1. Set the number of time steps into the future, $t_f$
2. Initialize an empty matrix for the forecasted $X$ values - This will be a matrix because it will be a collection of vectors. Each vector will represent one sample forecast
3. Initialize two steps of each of the sample vectors to be generated - In practical terms, initialize the first number of each column; each _column_ will represent a forecast time series

Keep in mind that what will be done here is to create samples of the future behavior of the signal $t_f$ number of time steps into the future. To do this, we will generate signals that use the posterior distributions of the parameters $\theta$ by calling the function `rand(chain[:,Z,Z])` which will randomly pick a number out of the sample pool, effectively "sampling" from that posterior distribution (sample pool).



```julia
time_future = 15

X_future = Matrix{Float64}(undef, time_future+2, samples)

# Initialize the first two time steps for every forecast
X_future[1, :] .= X[time-1]
X_future[2, :] .= X[time]

# populate the forecast vectors by sampling from the posterior sample pool of the parameters θ

for col in 1:samples
	phi_1_future = rand(chain[:,1,1])
	phi_2_future = rand(chain[:,2,1])
	error_future = rand(chain[:,3,1])
	noise_future = rand(Normal(0, error_future))
		
	for row in 3:(time_future+2)
		X_future[row, col] = 
			phi_1_future * X_future[row-1, col] + 
			phi_2_future * X_future[row-2, col] +
			noise_future
	end
end
```

#### Visualize the forecast

Now that we _propagated the uncertainty_ of in the posterior distribution of the parameters $\theta$, we can plot the posterior predictive distribution of $X$, $P(X^*|\theta)$.


```julia
time_predict = time:(time + time_future)

for i in 1:samples
	plot!(p_data, time_predict, X_future[2:end, i],
	legend = false,
	# predictions
	linewidth = 1, color = :green, alpha = 0.1
	)
end

p_data

# visualize mean values for predictions

X_future_mean = [mean(X_future[i, 1:samples]) for i in 2:(time_future+2)]

plot!(p_data, time_predict, X_future_mean, 
	legend = false,
	linewidth = 2, 
	color = :red, 
	linestyle = :dot
)
```


![svg](/images/20240222_Bayesian_Time_Series_Analysis/output_15_0.svg)



```julia

```

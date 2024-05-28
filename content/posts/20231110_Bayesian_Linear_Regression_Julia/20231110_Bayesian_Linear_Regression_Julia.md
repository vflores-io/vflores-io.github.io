+++
title = "Bayesian Linear Regression with Julia and Turing.jl"
date = 2023-11-10T14:53:29+08:00
draft = false
summary = " Learn the basics of Bayesian linear regression using Julia and Turing.jl. This tutorial covers model formulation, implementation, and interpretation through a practical example."
tags = ["Bayesian", "Linear", "Regression", "Turing", "Julia"]
+++
---

## Finding a Linear Relationship Between Height and Weight Using Bayesian Methods

### Problem Statement

You have some data on the relationship between the height and weight of some people, and you want to fit a linear model of the form:

$$y = \alpha + \beta x + \varepsilon$$

where $y$ is the weight, $x$ is the height, $\alpha$ is the intercept, $\beta$ is the slope, and $\varepsilon$ is the error term. You want to use Bayesian inference to estimate the posterior distributions of $\alpha$ and $\beta$ given the data and some prior assumptions. You also want to use probabilistic programming to implement the Bayesian model and perform inference using a package like `Turing.jl`.

Your task is to write the code in Julia that can generate some synthetic data (or use an existing data set), define the Bayesian linear regression model, and sample from the posterior distributions using Hamiltonian Monte Carlo (HMC).


###### Credit

This exercise is heavily inspired, and mostly taken from, the doggo's tutorial. Please visit his [Youtube channel here](https://www.youtube.com/@doggodotjl), it's an amazing starting point for Julia programming!


### Import the Necessary Packages


```julia
using LinearAlgebra, Turing, CSV, DataFrames, Plots, StatsPlots, LaTeXStrings
```

### Bayesian Workflow

For this exercise, I will implement the following workflow:
* Collect data: this will be implemented by downloading the relevant data
* Build a Bayesian model: will use `Turing.jl` to build the model
* Infer the posterior distributions of the parameters $\alpha$ and $\beta$
* Evaluate the fit of the model

#### Collecting the data

The data to be analyzed will be the height vs. weight data from: 
[https://www.kaggle.com/datasets/burnoutminer/heights-and-weights-dataset](https://www.kaggle.com/datasets/burnoutminer/heights-and-weights-dataset).

Since the dataset is too large, we will select only the first 1000 entries.


```julia
# collect data
# this data set was downloaded from kaggle:
# https://www.kaggle.com/datasets/burnoutminer/heights-and-weights-dataset

df = CSV.read(joinpath("data", "SOCR-HeightWeight.csv"), DataFrame)

# select only 100 entries
df = df[1:1000, :]

first(df, 5)
```




<div><div style = "float: left;"><span>5×3 DataFrame</span></div><div style = "clear: both;"></div></div><div class = "data-frame" style = "overflow-x: scroll;"><table class = "data-frame" style = "margin-bottom: 6px;"><thead><tr class = "header"><th class = "rowNumber" style = "font-weight: bold; text-align: right;">Row</th><th style = "text-align: left;">Index</th><th style = "text-align: left;">Height(Inches)</th><th style = "text-align: left;">Weight(Pounds)</th></tr><tr class = "subheader headerLastRow"><th class = "rowNumber" style = "font-weight: bold; text-align: right;"></th><th title = "Int64" style = "text-align: left;">Int64</th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Float64" style = "text-align: left;">Float64</th></tr></thead><tbody><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">1</td><td style = "text-align: right;">1</td><td style = "text-align: right;">65.7833</td><td style = "text-align: right;">112.993</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">2</td><td style = "text-align: right;">2</td><td style = "text-align: right;">71.5152</td><td style = "text-align: right;">136.487</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">3</td><td style = "text-align: right;">3</td><td style = "text-align: right;">69.3987</td><td style = "text-align: right;">153.027</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">4</td><td style = "text-align: right;">4</td><td style = "text-align: right;">68.2166</td><td style = "text-align: right;">142.335</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">5</td><td style = "text-align: right;">5</td><td style = "text-align: right;">67.7878</td><td style = "text-align: right;">144.297</td></tr></tbody></table></div>




```julia
# change the column headers for easier access

colnames = ["index","height","weight"]; rename!(df, Symbol.(colnames))

first(df, 5)
```




<div><div style = "float: left;"><span>5×3 DataFrame</span></div><div style = "clear: both;"></div></div><div class = "data-frame" style = "overflow-x: scroll;"><table class = "data-frame" style = "margin-bottom: 6px;"><thead><tr class = "header"><th class = "rowNumber" style = "font-weight: bold; text-align: right;">Row</th><th style = "text-align: left;">index</th><th style = "text-align: left;">height</th><th style = "text-align: left;">weight</th></tr><tr class = "subheader headerLastRow"><th class = "rowNumber" style = "font-weight: bold; text-align: right;"></th><th title = "Int64" style = "text-align: left;">Int64</th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Float64" style = "text-align: left;">Float64</th></tr></thead><tbody><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">1</td><td style = "text-align: right;">1</td><td style = "text-align: right;">65.7833</td><td style = "text-align: right;">112.993</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">2</td><td style = "text-align: right;">2</td><td style = "text-align: right;">71.5152</td><td style = "text-align: right;">136.487</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">3</td><td style = "text-align: right;">3</td><td style = "text-align: right;">69.3987</td><td style = "text-align: right;">153.027</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">4</td><td style = "text-align: right;">4</td><td style = "text-align: right;">68.2166</td><td style = "text-align: right;">142.335</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">5</td><td style = "text-align: right;">5</td><td style = "text-align: right;">67.7878</td><td style = "text-align: right;">144.297</td></tr></tbody></table></div>



#### Visualizing the Data



```julia
plot_data = scatter(df.height, df.weight,
	legend = false,
	title = "Height vs. Weight",
	xlabel = "Height (in)",
	ylabel = "Weight (lb)"
)
```

![svg](/images/20231110_Bayesian_Linear_Regression_Julia/output_9_0.svg)



#### Building a Bayesian model with `Turing.jl`.

First, we assume that the weight is a variable dependent on the height. Thus, we can express the Bayesian model as:

$$y\sim N(\alpha + \beta^{T}\mathbf{X}, \sigma^2)$$

The above means that we assume that the data follows a normal distribution (in this case, a multivariate normal distribution), whose standard deviation is σ and its mean is the linear relationship $\alpha + \beta^{T}\mathbf{X}$.

Next, we need to assign priors to the variables $\alpha$, $\beta$ and $\sigma^2$. The latter is a measure of the uncertainty in _the model_.

So, the priors will be assigned as follows:

$$\alpha \sim N(0,10)$$
$$\beta \sim U(0,50)$$
$$\sigma^{2} \sim TN(0,100;0,\infty)$$

The last distribution is a _truncated normal distribution_ bounded from 0 to $\infty$.


```julia
@model function blr(height, weight)

	# priors:
	α ~ Normal(0,10) # intercept
	β ~ Uniform(0,50)
	σ ~ truncated(Normal(0, 100); lower=0)  # variance standard distribution

	# likelihood
	# the likelihood in this case means that I assume that the data follows a
	# multivariate normal distribution, whose uncertainty is σ, and its mean is the linear relationship:
	avg_weight = α .+ (β.*height)

	# build the model
	weight ~ MvNormal(avg_weight, σ)
end
```




    blr (generic function with 2 methods)



The next step is to perform Bayesian inference. *Crank up the Bayes!*


```julia
# crank up the bayes!
model = blr(df.height, df.weight)
samples = 1000
chain = sample(model, NUTS(), samples)
```

    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39mFound initial step size
    [36m[1m└ [22m[39m  ϵ = 9.765625e-5
    [32mSampling: 100%|█████████████████████████████████████████| Time: 0:00:11[39m9m
    




    Chains MCMC chain (1000×15×1 Array{Float64, 3}):
    
    Iterations        = 501:1:1500
    Number of chains  = 1
    Samples per chain = 1000
    Wall duration     = 31.4 seconds
    Compute duration  = 31.4 seconds
    parameters        = α, β, σ
    internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size
    
    Summary Statistics
     [1m parameters [0m [1m     mean [0m [1m     std [0m [1m    mcse [0m [1m ess_bulk [0m [1m ess_tail [0m [1m    rhat [0m [1m [0m ⋯
     [90m     Symbol [0m [90m  Float64 [0m [90m Float64 [0m [90m Float64 [0m [90m  Float64 [0m [90m  Float64 [0m [90m Float64 [0m [90m [0m ⋯
    
               α   -34.8414    7.6414    0.4117   344.5155   365.1189    1.0038    ⋯
               β     2.3859    0.1124    0.0060   345.5269   345.0618    1.0039    ⋯
               σ    10.3030    0.2239    0.0100   509.4680   389.9078    1.0016    ⋯
    [36m                                                                1 column omitted[0m
    
    Quantiles
     [1m parameters [0m [1m     2.5% [0m [1m    25.0% [0m [1m    50.0% [0m [1m    75.0% [0m [1m    97.5% [0m
     [90m     Symbol [0m [90m  Float64 [0m [90m  Float64 [0m [90m  Float64 [0m [90m  Float64 [0m [90m  Float64 [0m
    
               α   -49.8948   -39.7950   -34.9188   -29.8116   -19.8403
               β     2.1673     2.3108     2.3872     2.4580     2.6100
               σ     9.8649    10.1550    10.3018    10.4554    10.7449
    



#### Visualizing the MCMC Diagnostics and Summarizing the Results

Now that we have performed Bayesian inference using the `NUTS()` algorithm, we can visualize the results. Addisionally, call for a summary of the statistics of the inferred posterior distributions of $\theta$.


```julia
summarize(chain)
```




    
     [1m parameters [0m [1m     mean [0m [1m     std [0m [1m    mcse [0m [1m ess_bulk [0m [1m ess_tail [0m [1m    rhat [0m [1m [0m ⋯
     [90m     Symbol [0m [90m  Float64 [0m [90m Float64 [0m [90m Float64 [0m [90m  Float64 [0m [90m  Float64 [0m [90m Float64 [0m [90m [0m ⋯
    
               α   -34.8414    7.6414    0.4117   344.5155   365.1189    1.0038    ⋯
               β     2.3859    0.1124    0.0060   345.5269   345.0618    1.0039    ⋯
               σ    10.3030    0.2239    0.0100   509.4680   389.9078    1.0016    ⋯
    [36m                                                                1 column omitted[0m
    




```julia
plot(chain)
```

![svg](/images/20231110_Bayesian_Linear_Regression_Julia/output_16_0.svg)




##### Visualizing the results

It is worth noting that the results from a Bayesian Linear Regression is not one single regression line, but many. From PyMC's [Generalized Linear Regression tutorial](https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/GLM_linear.html):

> In GLMs, we do not only have one best fitting regression line, but many. A posterior predictive plot takes multiple samples from the posterior (intercepts and slopes) and plots a regression line for each of them. We can manually generate these regression lines using the posterior samples directly.

What this means is that if we want to visualize all the lines that are generated by the parameter posterior distribution sample pool, we need to generate one line per sample set, and then we can plot them all. This procedure is executed next.


```julia
# plot all the sample regressions
# this method was taken from: https://www.youtube.com/watch?v=EgrrtZEVOv0&t=1113s

for i in 1:samples
	α = chain[i,1,1]    #chain[row, column, chain_ID]
	β = chain[i,2,1]
	σ² = chain[i,3,1]
	plot!(plot_data, x -> α + β*x,
		legend = false,
		# samples
		linewidth = 2, color = :orange, alpha = 0.02,
		# error
        ribbon = σ², fillalpha = 0.002
    )
end	

plot_data
```


![svg](/images/20231110_Bayesian_Linear_Regression_Julia/output_18_0.svg)






### Using the Regression Model to Make Predictions

Select the heights for which we want to predict the weights and then run the prediction command from `Turing`.


```julia
pred_height = [62, 84, 75, 70, 71, 67]
predictions = predict(blr(pred_height, missing), chain)
```




    Chains MCMC chain (1000×6×1 Array{Float64, 3}):
    
    Iterations        = 1:1:1000
    Number of chains  = 1
    Samples per chain = 1000
    parameters        = weight[1], weight[2], weight[3], weight[4], weight[5], weight[6]
    internals         = 
    
    Summary Statistics
     [1m parameters [0m [1m     mean [0m [1m     std [0m [1m    mcse [0m [1m  ess_bulk [0m [1m ess_tail [0m [1m    rhat [0m [1m[0m ⋯
     [90m     Symbol [0m [90m  Float64 [0m [90m Float64 [0m [90m Float64 [0m [90m   Float64 [0m [90m  Float64 [0m [90m Float64 [0m [90m[0m ⋯
    
       weight[1]   113.6815   10.3344    0.3270    997.5393   947.2109    0.9993   ⋯
       weight[2]   165.3164   10.8352    0.3744    832.5405   818.6640    1.0008   ⋯
       weight[3]   143.8911   10.5355    0.3461    929.5467   874.2977    0.9993   ⋯
       weight[4]   132.3417   10.4836    0.3448    921.6347   943.0320    1.0007   ⋯
       weight[5]   134.7606   10.7046    0.3350   1023.8876   977.6814    1.0025   ⋯
       weight[6]   124.9423   10.2245    0.3247    993.9282   867.7391    0.9991   ⋯
    [36m                                                                1 column omitted[0m
    
    Quantiles
     [1m parameters [0m [1m     2.5% [0m [1m    25.0% [0m [1m    50.0% [0m [1m    75.0% [0m [1m    97.5% [0m
     [90m     Symbol [0m [90m  Float64 [0m [90m  Float64 [0m [90m  Float64 [0m [90m  Float64 [0m [90m  Float64 [0m
    
       weight[1]    93.9378   106.3972   113.6943   120.8093   134.9264
       weight[2]   142.4871   158.4933   165.5406   172.7313   184.7437
       weight[3]   122.8292   137.0108   144.0339   151.1920   164.2645
       weight[4]   111.8872   125.3733   132.1726   139.2690   153.7222
       weight[5]   113.9147   127.4356   135.0149   142.1375   154.5537
       weight[6]   105.3221   118.0098   125.1640   131.6011   145.2976
    



#### Visualize the Distributions of the Predicted Weights


```julia
plot(predictions)
```
![svg](/images/20231110_Bayesian_Linear_Regression_Julia/output_22_0.svg)


Finally, to obtain a point estimate, compute the mean weight prediction for each height.


```julia
mean_predictions = mean(predictions)
```




    Mean
     [1m parameters [0m [1m     mean [0m
     [90m     Symbol [0m [90m  Float64 [0m
    
       weight[1]   113.6815
       weight[2]   165.3164
       weight[3]   143.8911
       weight[4]   132.3417
       weight[5]   134.7606
       weight[6]   124.9423
    



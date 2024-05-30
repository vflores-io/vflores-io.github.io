+++
title = 'Bayesian Poisson Regression with Julia and Turing.jl'
date = 2024-02-17T11:57:07+08:00
draft = false
summary = "Explore Bayesian Poisson regression for modeling count data with Julia and Turing.jl. This tutorial includes model setup, implementation, and performance assessment with a practical example."
tags = ["Bayesian", "Bayesian Regression", "Poisson", "Regression", "Turing", "Julia"]
+++
---

In this example, I am following the tutorials found in:
- [Turing.jl - Bayesian Poisson Regression](https://turinglang.org/dev/tutorials/07-poisson-regression/)
- [PyMC - GLM: Poisson Regression](https://www.pymc.io/projects/examples/en/latest/generalized_linear_models/GLM-poisson-regression.html)

Both examples show the interaction between some variables and a discrete outcome. In this case, the outcome is the number of sneezes per day (i.e. a discrete outcome) in some study subjects, and whether or not they take antihistamine medicine and whether or not they drink alcohol. 

This example explores how these factors, and more specifically, the combination of these factors, affect the number of times a person sneezes.


```julia
using CSV, DataFrames, Turing, StatsPlots, Plots, Random
Random.seed!(42)
```




    TaskLocalRNG()



## Collect (generate) the data

In this example, we will generate the data in the same way as in the tutorials:

|             	| No Alcohol 	| Alcohol 	|
|:-------------|:------------:|:---------:|
| **No Meds** 	| 6          	| 36      	|
| **Meds**    	| 1          	| 3       	|

Those values will be used to create the artificial data by generating Poisson-distributed random samples.



```julia
theta_noalc_nomed = 6
theta_noalc_med = 1
theta_alc_nomed = 36
theta_alc_med = 3

ns = 500    # number of samples

# create a data frame

data = DataFrame(
    hcat(
        vcat(
        rand(Poisson(theta_noalc_med), ns),
        rand(Poisson(theta_alc_med), ns),
        rand(Poisson(theta_noalc_nomed), ns),
        rand(Poisson(theta_alc_nomed), ns)
        ),
        vcat(
            falses(ns),
            trues(ns),
            falses(ns),
            trues(ns)
        ),
        vcat(
            falses(ns),
            falses(ns),
            trues(ns),
            trues(ns)
        )
    ), :auto
)

# assign names to headers

head_names = [:n_sneezes, :alcohol, :nomeds]

sneeze_data = DataFrame(data, head_names)

first(sneeze_data, 10)
```




<div><div style = "float: left;"><span>10×3 DataFrame</span></div><div style = "clear: both;"></div></div><div class = "data-frame" style = "overflow-x: scroll;"><table class = "data-frame" style = "margin-bottom: 6px;"><thead><tr class = "header"><th class = "rowNumber" style = "font-weight: bold; text-align: right;">Row</th><th style = "text-align: left;">n_sneezes</th><th style = "text-align: left;">alcohol</th><th style = "text-align: left;">nomeds</th></tr><tr class = "subheader headerLastRow"><th class = "rowNumber" style = "font-weight: bold; text-align: right;"></th><th title = "Int64" style = "text-align: left;">Int64</th><th title = "Int64" style = "text-align: left;">Int64</th><th title = "Int64" style = "text-align: left;">Int64</th></tr></thead><tbody><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">1</td><td style = "text-align: right;">0</td><td style = "text-align: right;">0</td><td style = "text-align: right;">0</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">2</td><td style = "text-align: right;">1</td><td style = "text-align: right;">0</td><td style = "text-align: right;">0</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">3</td><td style = "text-align: right;">0</td><td style = "text-align: right;">0</td><td style = "text-align: right;">0</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">4</td><td style = "text-align: right;">0</td><td style = "text-align: right;">0</td><td style = "text-align: right;">0</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">5</td><td style = "text-align: right;">1</td><td style = "text-align: right;">0</td><td style = "text-align: right;">0</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">6</td><td style = "text-align: right;">1</td><td style = "text-align: right;">0</td><td style = "text-align: right;">0</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">7</td><td style = "text-align: right;">1</td><td style = "text-align: right;">0</td><td style = "text-align: right;">0</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">8</td><td style = "text-align: right;">1</td><td style = "text-align: right;">0</td><td style = "text-align: right;">0</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">9</td><td style = "text-align: right;">2</td><td style = "text-align: right;">0</td><td style = "text-align: right;">0</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">10</td><td style = "text-align: right;">2</td><td style = "text-align: right;">0</td><td style = "text-align: right;">0</td></tr></tbody></table></div>



### Visualize the data

Now that we have "collected" some data on the number of sneezes per day from a number of people, we visualize the data.

The way we are collecting and plotting these data sub-sets is as follows:
1. Call the histogram function
2. Create a histogram of the dataframe "sneeze_data" we "collected" previously
3. Select a subset of that dataframe
4. All the rows of the columns where alcohol is `false` i.e. 0 AND all the rows where no medicine was taken is also `false`
5. All the rows of the columns where alcohol is `false` AND all the rows of where medicine is `true`
6. ... and so on


```julia
# create separate histograms for each case

p1 = histogram(sneeze_data[(sneeze_data[:,:alcohol] .== 0) .& (sneeze_data[:,:nomeds] .== 0), :n_sneezes]; title = "No alcohol + No Meds", ylabel="People Count")
p2 = histogram(sneeze_data[(sneeze_data[:,:alcohol] .== 1) .& (sneeze_data[:,:nomeds] .== 0), :n_sneezes]; title = "No alcohol + Meds")
p3 = histogram(sneeze_data[(sneeze_data[:,:alcohol] .== 0) .& (sneeze_data[:,:nomeds] .== 1), :n_sneezes]; title = "Alcohol + No Meds", xlabel = "Sneezes/Day", ylabel="People Count")
p4 = histogram(sneeze_data[(sneeze_data[:,:alcohol] .== 1) .& (sneeze_data[:,:nomeds] .== 1), :n_sneezes]; title = "Alcohol + Meds", xlabel = "Sneezes/Day")

plot(p1, p2, p3, p4; layout=(2,2), legend = false)
```
![svg](/images/20240217_Bayesian_Poisson_Regression/output_5_0.svg)


### Interpreting the data

The histograms show that the data from the "study" resembles a Poisson distribution (as mentioned in the PyMC tutorial, this is obvious, because that's how the data is generated!). Furthermore, the data is telling us something:
- Looking at the plot for "no alcohol and medicine" it is clear that most people reported very few sneezes; notice how the histogram skews towards large counts (of people) for very few sneezes
- On the other hand, notice how the "alcohol and _no_ medicine" seems to tell us that many reported somewhere around 35 sneezes per day

Again, we can start thinking of a pattern just by looking at the data, and it seems like the data is telling us that if you don't drink alcohol and take antihistamines, you are less likely to be sneezing around than if you drink alcohol and don't take any allergy meds. Makes sense, right?

## Model

We established that the data looks like it could be modelled as a Poisson distribution. Thus, we can define our probabilistic model as follows:

$$Y_{obs} \sim Poisson(\lambda)$$

$$\log(\lambda) = \theta'\mathbf{x} = \alpha + \beta' \mathbf{x}$$

What the above means is that we assume that the observed data outcomes, i.e., the number of sneezes per day, follow a Poisson distribution, which is a discrete probability distribution that models the number of events that occur in a fixed interval of time or space. The rate or intensity of the events, $\lambda$, depends on the predictor variables (the input data) $\mathcal{x}$, such as the season, the temperature, or, in our case, whether a person ingested alcohol and whether the person took antihistamines.

The linear predictor $\theta' \mathcal{x}$ is the function that links the predictor variables to the rate parameter, where $\theta = \{\alpha, \beta'\}$ are the parameters of the model. 

Looking at the structure of the linear relationship between the paramters of the model, and the predictors:

$$\log(\lambda) = \alpha + \beta' \mathcal{x}$$

we can understand that the parameter $\alpha$ is the intercept, which is the expected number of sneezes when all the predictor variables are zero. The parameter $\beta'$ is a vector of coefficients, which measure the effect of each predictor variable $\mathcal{x}$ on the number of sneezes. The log link function ensures that the rate parameter $\lambda$ is always positive and allows for multiplicative effects of the predictor variables on the response variable.

### Define the model with `Turing.jl`

Now that we know how we are modeling our data, we use the package `Turing.jl` to define the model. `Turing.jl` is a tool that helps us write models in Julia and find the best parameters for them.

The model has two parts: the prior and the likelihood. The prior is what we think or guess about the parameters before we see the data. The likelihood is how likely the data is under the parameters. The parameters are the numbers that control the model, such as the rate of sneezes.

We use the Poisson distribution for the likelihood, because it is good for counting things, like sneezes. The Poisson distribution has one parameter, the rate of sneezes. The higher the rate, the more sneezes we expect.

We use any distribution for the prior, depending on how much we know about the parameters. If we know nothing, we use a flat prior, which does not favor any value. The prior affects the final answer, because it is our starting point.

We use Bayes’ theorem to combine the prior and the likelihood and get the final answer. The final answer is the posterior, which is what we believe about the parameters after we see the data. The posterior is the best fit for the model and the data.

**Let's crank up the Bayes!**


```julia
@model function poisson(x, y)

		# define the priors
		alpha ~ Normal(0,1)
		alcohol ~ Normal(0, 1)
		nomeds ~ Normal(0, 1)
		# alc_med ~ Normal(0,1)

		# define the likelihood
		for i in 1:length(y)
	        log_lambda = alpha + alcohol * x[i, 1] + nomeds * x[i, 2] 
	        lambda = exp(log_lambda)
	        y[i] ~ Poisson(lambda)
	    end
	
	end
```




    poisson (generic function with 2 methods)




```julia
# pass the data to the model function
	# pass the predictor data as a Matrix for efficiency
model = poisson(Matrix(sneeze_data[!,[:alcohol, :nomeds] ]), sneeze_data[!, :n_sneezes])

# select the sampler
sampler = NUTS()

# define the number of sampler
samples = 1000

# set number of chains
num_chains = 8
	
# crank up the Bayes!
chain = sample(model, sampler, MCMCThreads(), samples, num_chains)
```

    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39mFound initial step size
    [36m[1m└ [22m[39m  ϵ = 0.00625
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39mFound initial step size
    [36m[1m└ [22m[39m  ϵ = 0.0125
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39mFound initial step size
    [36m[1m└ [22m[39m  ϵ = 0.00625
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39mFound initial step size
    [36m[1m└ [22m[39m  ϵ = 0.0125
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39mFound initial step size
    [36m[1m└ [22m[39m  ϵ = 0.0125
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39mFound initial step size
    [36m[1m└ [22m[39m  ϵ = 0.00625
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39mFound initial step size
    [36m[1m└ [22m[39m  ϵ = 0.00625
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39mFound initial step size
    [36m[1m└ [22m[39m  ϵ = 0.0125
    [32mSampling (8 threads): 100%|█████████████████████████████| Time: 0:00:00[39m
    




    Chains MCMC chain (1000×15×8 Array{Float64, 3}):
    
    Iterations        = 501:1:1500
    Number of chains  = 8
    Samples per chain = 1000
    Wall duration     = 13.66 seconds
    Compute duration  = 100.67 seconds
    parameters        = alpha, alcohol, nomeds
    internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size
    
    Summary Statistics
     [1m parameters [0m [1m    mean [0m [1m     std [0m [1m    mcse [0m [1m  ess_bulk [0m [1m  ess_tail [0m [1m    rhat [0m [1m[0m ⋯
     [90m     Symbol [0m [90m Float64 [0m [90m Float64 [0m [90m Float64 [0m [90m   Float64 [0m [90m   Float64 [0m [90m Float64 [0m [90m[0m ⋯
    
           alpha   -0.5025    0.0277    0.0005   2943.5608   2841.2874    1.0030   ⋯
         alcohol    1.7333    0.0186    0.0003   3801.1996   3652.2403    1.0022   ⋯
          nomeds    2.3348    0.0236    0.0004   2901.3750   3410.6453    1.0020   ⋯
    [36m                                                                1 column omitted[0m
    
    Quantiles
     [1m parameters [0m [1m    2.5% [0m [1m   25.0% [0m [1m   50.0% [0m [1m   75.0% [0m [1m   97.5% [0m
     [90m     Symbol [0m [90m Float64 [0m [90m Float64 [0m [90m Float64 [0m [90m Float64 [0m [90m Float64 [0m
    
           alpha   -0.5568   -0.5212   -0.5023   -0.4839   -0.4486
         alcohol    1.6974    1.7205    1.7331    1.7458    1.7698
          nomeds    2.2891    2.3189    2.3346    2.3506    2.3824
    



**NOTE:** The above routine employs the MCMCThreads method to sample multiple chains. However, in order to implement this, one needs to change the environment variables for the number of threads Julia can use. These two threads might shed some light as to how to achieve this:
1. [https://docs.julialang.org/en/v1/manual/multi-threading/#man-multithreading](https://docs.julialang.org/en/v1/manual/multi-threading/#man-multithreading)
2. [https://discourse.julialang.org/t/julia-num-threads-in-vs-code-windows-10-wsl/28794](https://discourse.julialang.org/t/julia-num-threads-in-vs-code-windows-10-wsl/28794)

Of course, if you don't want to bother, then just change the last two functional lines in the cell above so that they read:

```julia
	# set number of chains - comment this out:
	# num_chains = 8
	
	# crank up the Bayes! - delete MCMCThreads() and num_chains
	chain = sample(model, sampler, samples)
``` 

### Visualize the results

We can see above that we have obtained a sample pool of the posterior distribution of the parameters. This is what we were looking for. What this means is that now we have a posterior distribution (in the form of a sample pool), which we can also summarize with summary statistics.

Let's look at the diagnostics plots and the summary statistics.


```julia
plot(chain)
```
![svg](/images/20240217_Bayesian_Poisson_Regression/output_13_0.svg)


```julia
DataFrame(summarystats(chain))
```




<div><div style = "float: left;"><span>3×8 DataFrame</span></div><div style = "clear: both;"></div></div><div class = "data-frame" style = "overflow-x: scroll;"><table class = "data-frame" style = "margin-bottom: 6px;"><thead><tr class = "header"><th class = "rowNumber" style = "font-weight: bold; text-align: right;">Row</th><th style = "text-align: left;">parameters</th><th style = "text-align: left;">mean</th><th style = "text-align: left;">std</th><th style = "text-align: left;">mcse</th><th style = "text-align: left;">ess_bulk</th><th style = "text-align: left;">ess_tail</th><th style = "text-align: left;">rhat</th><th style = "text-align: left;">ess_per_sec</th></tr><tr class = "subheader headerLastRow"><th class = "rowNumber" style = "font-weight: bold; text-align: right;"></th><th title = "Symbol" style = "text-align: left;">Symbol</th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Float64" style = "text-align: left;">Float64</th></tr></thead><tbody><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">1</td><td style = "text-align: left;">alpha</td><td style = "text-align: right;">-0.502519</td><td style = "text-align: right;">0.0276553</td><td style = "text-align: right;">0.000511069</td><td style = "text-align: right;">2943.56</td><td style = "text-align: right;">2841.29</td><td style = "text-align: right;">1.00298</td><td style = "text-align: right;">29.2397</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">2</td><td style = "text-align: left;">alcohol</td><td style = "text-align: right;">1.7333</td><td style = "text-align: right;">0.0186097</td><td style = "text-align: right;">0.000301611</td><td style = "text-align: right;">3801.2</td><td style = "text-align: right;">3652.24</td><td style = "text-align: right;">1.00224</td><td style = "text-align: right;">37.759</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">3</td><td style = "text-align: left;">nomeds</td><td style = "text-align: right;">2.3348</td><td style = "text-align: right;">0.0236269</td><td style = "text-align: right;">0.000436385</td><td style = "text-align: right;">2901.38</td><td style = "text-align: right;">3410.65</td><td style = "text-align: right;">1.00197</td><td style = "text-align: right;">28.8207</td></tr></tbody></table></div>




```julia
# taking the first chain
c1 = chain[:, :, 1]

# Calculating the exponentiated means
b0_exp = exp(mean(c1[:alpha]))
b1_exp = exp(mean(c1[:alcohol]))
b2_exp = exp(mean(c1[:nomeds]))

println("The exponent of the mean of the weights (or coefficients) are: \n")
println("b0: ", b0_exp)
println("b1: ", b1_exp)
println("b2: ", b2_exp)
```

    The exponent of the mean of the weights (or coefficients) are: 
    
    b0: 0.604415461752317
    b1: 5.658573583760772
    b2: 10.342642711232362
    

Notice how we are **not** recovering the original $\lambda$ values that were used to create this data set, i.e.:

```julia
	theta_noalc_nomed = 6
	theta_noalc_med = 1
	theta_alc_nomed = 36
	theta_alc_med = 3
```
Instead, we are recovering _the parameters of the linear function_, in other words, $\theta = \{\alpha, \beta'\}$ in the linear relation:

$$\log(\lambda) = \alpha + \beta_1 x_{alc} + \beta_2 x_{meds}$$

where $x_{(\cdot)}$ represents the binary variable of whether the subject took alcohol/medicine or not.

## Conclusion

This tutorial shows how to perform Bayesian inference on _discrete_ data, e.g. the record of how many sneezes per day a group of people had, and classified according to their alcohol and medication consumption. 

In real-world scenarios, we would obviously not know the parameter values, since this is precisely what we want to find out by incorporating whatever we knew about them into what we observed.

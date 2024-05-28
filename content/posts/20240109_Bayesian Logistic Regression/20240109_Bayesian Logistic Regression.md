+++
title = 'Bayesian Logistic Regression with Julia and Turing.jl'
date = 2024-01-09T11:57:07+08:00
draft = false
summary = "Applying Turing.jl package in Julia for a probabilistic approach to a classification problem on a real-world dataset."
tags = [ "Bayesian", "Logistic", "Regression", "Turing", "Julia"]
+++
---

## Problem Statement

You are interested in studying the factors that influence the likelihood of heart disease among patients. 

You have a dataset of 303 patients, each with 14 variables: age, sex, chest pain type, resting blood pressure, serum cholesterol, fasting blood sugar, resting electrocardiographic results, maximum heart rate achieved, exercise induced angina, oldpeak, slope, number of major vessels, thalassemia, and diagnosis of heart disease. 

You want to use Bayesian logistic regression to model the probability of heart disease (the outcome variable) as a function of some or all of the other variables (the predictor variables). 

You also want to compare different models and assess their fit and predictive performance.

## Bayesian Workflow

For this project, I will try to follow this workflow:

1. Data exploration: Explore the data using descriptive statistics and visualizations to get a sense of the distribution, range, and correlation of the variables. Identify any outliers, missing values, or potential errors in the data. Transform or standardize the variables if needed.

2. Model specification: Specify a probabilistic model that relates the outcome variable to the predictor variables using a logistic regression equation. Choose appropriate priors for the model parameters, such as normal, student-t, or Cauchy distributions. You can use the `brms` package in Julia to define and fit Bayesian models using a formula syntax similar to `lme4`. However, try to use `Turing.jl`

3. Model fitting: Fit the model using a sampling algorithm such as Hamiltonian Monte Carlo (HMC) or No-U-Turn Sampler (NUTS). You can use the `DynamicHMC` or `Turing.jl` package in Julia to implement these algorithms. Check the convergence and mixing of the chains using diagnostics such as trace plots, autocorrelation plots, effective sample size, and potential scale reduction factor. You can use the `MCMCDiagnostics` or the included diagnostics features in `Turing.jl` package in Julia to compute these diagnostics.

4. Model checking: Check the fit and validity of the model using posterior predictive checks, residual analysis, and sensitivity analysis. You can use the `PPCheck` package in Julia to perform posterior predictive checks, which compare the observed data to data simulated from the posterior predictive distribution. You can use the `BayesianRidgeRegression` package in Julia to perform residual analysis, which plots the residuals against the fitted values and the predictor variables. You can use the `Sensitivity` package in Julia to perform sensitivity analysis, which measures how the posterior distribution changes with respect to the prior distribution or the likelihood function.


```julia
# import packages
using CSV, Turing, DataFrames, StatsPlots, LaTeXStrings, Distributions
using Images, ImageIO
using Random: seed!
seed!(42)
```




    Random.TaskLocalRNG()



## Data Exploration

After "collecting" the data, we may import it and arrange it so we can use it further.

The data set can be found in this [Kaggle link](https://www.kaggle.com/datasets/aavigan/cleveland-clinic-heart-disease-dataset).


```julia
df = CSV.read("data/processed_cleveland.csv", DataFrame)
map!(x -> x != 0 ? 1 : 0, df.num, df.num); # make the outcome binary
df
```




<div><div style = "float: left;"><span>303×14 DataFrame</span></div><div style = "float: right;"><span style = "font-style: italic;">278 rows omitted</span></div><div style = "clear: both;"></div></div><div class = "data-frame" style = "overflow-x: scroll;"><table class = "data-frame" style = "margin-bottom: 6px;"><thead><tr class = "header"><th class = "rowNumber" style = "font-weight: bold; text-align: right;">Row</th><th style = "text-align: left;">age</th><th style = "text-align: left;">sex</th><th style = "text-align: left;">cp</th><th style = "text-align: left;">trestbps</th><th style = "text-align: left;">chol</th><th style = "text-align: left;">fbs</th><th style = "text-align: left;">restecg</th><th style = "text-align: left;">thalach</th><th style = "text-align: left;">exang</th><th style = "text-align: left;">oldpeak</th><th style = "text-align: left;">slope</th><th style = "text-align: left;">ca</th><th style = "text-align: left;">thal</th><th style = "text-align: left;">num</th></tr><tr class = "subheader headerLastRow"><th class = "rowNumber" style = "font-weight: bold; text-align: right;"></th><th title = "Int64" style = "text-align: left;">Int64</th><th title = "Int64" style = "text-align: left;">Int64</th><th title = "Int64" style = "text-align: left;">Int64</th><th title = "Int64" style = "text-align: left;">Int64</th><th title = "Int64" style = "text-align: left;">Int64</th><th title = "Int64" style = "text-align: left;">Int64</th><th title = "Int64" style = "text-align: left;">Int64</th><th title = "Int64" style = "text-align: left;">Int64</th><th title = "Int64" style = "text-align: left;">Int64</th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Int64" style = "text-align: left;">Int64</th><th title = "String1" style = "text-align: left;">String1</th><th title = "String1" style = "text-align: left;">String1</th><th title = "Int64" style = "text-align: left;">Int64</th></tr></thead><tbody><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">1</td><td style = "text-align: right;">63</td><td style = "text-align: right;">1</td><td style = "text-align: right;">1</td><td style = "text-align: right;">145</td><td style = "text-align: right;">233</td><td style = "text-align: right;">1</td><td style = "text-align: right;">2</td><td style = "text-align: right;">150</td><td style = "text-align: right;">0</td><td style = "text-align: right;">2.3</td><td style = "text-align: right;">3</td><td style = "text-align: left;">0</td><td style = "text-align: left;">6</td><td style = "text-align: right;">0</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">2</td><td style = "text-align: right;">67</td><td style = "text-align: right;">1</td><td style = "text-align: right;">4</td><td style = "text-align: right;">160</td><td style = "text-align: right;">286</td><td style = "text-align: right;">0</td><td style = "text-align: right;">2</td><td style = "text-align: right;">108</td><td style = "text-align: right;">1</td><td style = "text-align: right;">1.5</td><td style = "text-align: right;">2</td><td style = "text-align: left;">3</td><td style = "text-align: left;">3</td><td style = "text-align: right;">1</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">3</td><td style = "text-align: right;">67</td><td style = "text-align: right;">1</td><td style = "text-align: right;">4</td><td style = "text-align: right;">120</td><td style = "text-align: right;">229</td><td style = "text-align: right;">0</td><td style = "text-align: right;">2</td><td style = "text-align: right;">129</td><td style = "text-align: right;">1</td><td style = "text-align: right;">2.6</td><td style = "text-align: right;">2</td><td style = "text-align: left;">2</td><td style = "text-align: left;">7</td><td style = "text-align: right;">1</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">4</td><td style = "text-align: right;">37</td><td style = "text-align: right;">1</td><td style = "text-align: right;">3</td><td style = "text-align: right;">130</td><td style = "text-align: right;">250</td><td style = "text-align: right;">0</td><td style = "text-align: right;">0</td><td style = "text-align: right;">187</td><td style = "text-align: right;">0</td><td style = "text-align: right;">3.5</td><td style = "text-align: right;">3</td><td style = "text-align: left;">0</td><td style = "text-align: left;">3</td><td style = "text-align: right;">0</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">5</td><td style = "text-align: right;">41</td><td style = "text-align: right;">0</td><td style = "text-align: right;">2</td><td style = "text-align: right;">130</td><td style = "text-align: right;">204</td><td style = "text-align: right;">0</td><td style = "text-align: right;">2</td><td style = "text-align: right;">172</td><td style = "text-align: right;">0</td><td style = "text-align: right;">1.4</td><td style = "text-align: right;">1</td><td style = "text-align: left;">0</td><td style = "text-align: left;">3</td><td style = "text-align: right;">0</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">6</td><td style = "text-align: right;">56</td><td style = "text-align: right;">1</td><td style = "text-align: right;">2</td><td style = "text-align: right;">120</td><td style = "text-align: right;">236</td><td style = "text-align: right;">0</td><td style = "text-align: right;">0</td><td style = "text-align: right;">178</td><td style = "text-align: right;">0</td><td style = "text-align: right;">0.8</td><td style = "text-align: right;">1</td><td style = "text-align: left;">0</td><td style = "text-align: left;">3</td><td style = "text-align: right;">0</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">7</td><td style = "text-align: right;">62</td><td style = "text-align: right;">0</td><td style = "text-align: right;">4</td><td style = "text-align: right;">140</td><td style = "text-align: right;">268</td><td style = "text-align: right;">0</td><td style = "text-align: right;">2</td><td style = "text-align: right;">160</td><td style = "text-align: right;">0</td><td style = "text-align: right;">3.6</td><td style = "text-align: right;">3</td><td style = "text-align: left;">2</td><td style = "text-align: left;">3</td><td style = "text-align: right;">1</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">8</td><td style = "text-align: right;">57</td><td style = "text-align: right;">0</td><td style = "text-align: right;">4</td><td style = "text-align: right;">120</td><td style = "text-align: right;">354</td><td style = "text-align: right;">0</td><td style = "text-align: right;">0</td><td style = "text-align: right;">163</td><td style = "text-align: right;">1</td><td style = "text-align: right;">0.6</td><td style = "text-align: right;">1</td><td style = "text-align: left;">0</td><td style = "text-align: left;">3</td><td style = "text-align: right;">0</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">9</td><td style = "text-align: right;">63</td><td style = "text-align: right;">1</td><td style = "text-align: right;">4</td><td style = "text-align: right;">130</td><td style = "text-align: right;">254</td><td style = "text-align: right;">0</td><td style = "text-align: right;">2</td><td style = "text-align: right;">147</td><td style = "text-align: right;">0</td><td style = "text-align: right;">1.4</td><td style = "text-align: right;">2</td><td style = "text-align: left;">1</td><td style = "text-align: left;">7</td><td style = "text-align: right;">1</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">10</td><td style = "text-align: right;">53</td><td style = "text-align: right;">1</td><td style = "text-align: right;">4</td><td style = "text-align: right;">140</td><td style = "text-align: right;">203</td><td style = "text-align: right;">1</td><td style = "text-align: right;">2</td><td style = "text-align: right;">155</td><td style = "text-align: right;">1</td><td style = "text-align: right;">3.1</td><td style = "text-align: right;">3</td><td style = "text-align: left;">0</td><td style = "text-align: left;">7</td><td style = "text-align: right;">1</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">11</td><td style = "text-align: right;">57</td><td style = "text-align: right;">1</td><td style = "text-align: right;">4</td><td style = "text-align: right;">140</td><td style = "text-align: right;">192</td><td style = "text-align: right;">0</td><td style = "text-align: right;">0</td><td style = "text-align: right;">148</td><td style = "text-align: right;">0</td><td style = "text-align: right;">0.4</td><td style = "text-align: right;">2</td><td style = "text-align: left;">0</td><td style = "text-align: left;">6</td><td style = "text-align: right;">0</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">12</td><td style = "text-align: right;">56</td><td style = "text-align: right;">0</td><td style = "text-align: right;">2</td><td style = "text-align: right;">140</td><td style = "text-align: right;">294</td><td style = "text-align: right;">0</td><td style = "text-align: right;">2</td><td style = "text-align: right;">153</td><td style = "text-align: right;">0</td><td style = "text-align: right;">1.3</td><td style = "text-align: right;">2</td><td style = "text-align: left;">0</td><td style = "text-align: left;">3</td><td style = "text-align: right;">0</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">13</td><td style = "text-align: right;">56</td><td style = "text-align: right;">1</td><td style = "text-align: right;">3</td><td style = "text-align: right;">130</td><td style = "text-align: right;">256</td><td style = "text-align: right;">1</td><td style = "text-align: right;">2</td><td style = "text-align: right;">142</td><td style = "text-align: right;">1</td><td style = "text-align: right;">0.6</td><td style = "text-align: right;">2</td><td style = "text-align: left;">1</td><td style = "text-align: left;">6</td><td style = "text-align: right;">1</td></tr><tr><td style = "text-align: right;">&vellip;</td><td style = "text-align: right;">&vellip;</td><td style = "text-align: right;">&vellip;</td><td style = "text-align: right;">&vellip;</td><td style = "text-align: right;">&vellip;</td><td style = "text-align: right;">&vellip;</td><td style = "text-align: right;">&vellip;</td><td style = "text-align: right;">&vellip;</td><td style = "text-align: right;">&vellip;</td><td style = "text-align: right;">&vellip;</td><td style = "text-align: right;">&vellip;</td><td style = "text-align: right;">&vellip;</td><td style = "text-align: right;">&vellip;</td><td style = "text-align: right;">&vellip;</td><td style = "text-align: right;">&vellip;</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">292</td><td style = "text-align: right;">55</td><td style = "text-align: right;">0</td><td style = "text-align: right;">2</td><td style = "text-align: right;">132</td><td style = "text-align: right;">342</td><td style = "text-align: right;">0</td><td style = "text-align: right;">0</td><td style = "text-align: right;">166</td><td style = "text-align: right;">0</td><td style = "text-align: right;">1.2</td><td style = "text-align: right;">1</td><td style = "text-align: left;">0</td><td style = "text-align: left;">3</td><td style = "text-align: right;">0</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">293</td><td style = "text-align: right;">44</td><td style = "text-align: right;">1</td><td style = "text-align: right;">4</td><td style = "text-align: right;">120</td><td style = "text-align: right;">169</td><td style = "text-align: right;">0</td><td style = "text-align: right;">0</td><td style = "text-align: right;">144</td><td style = "text-align: right;">1</td><td style = "text-align: right;">2.8</td><td style = "text-align: right;">3</td><td style = "text-align: left;">0</td><td style = "text-align: left;">6</td><td style = "text-align: right;">1</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">294</td><td style = "text-align: right;">63</td><td style = "text-align: right;">1</td><td style = "text-align: right;">4</td><td style = "text-align: right;">140</td><td style = "text-align: right;">187</td><td style = "text-align: right;">0</td><td style = "text-align: right;">2</td><td style = "text-align: right;">144</td><td style = "text-align: right;">1</td><td style = "text-align: right;">4.0</td><td style = "text-align: right;">1</td><td style = "text-align: left;">2</td><td style = "text-align: left;">7</td><td style = "text-align: right;">1</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">295</td><td style = "text-align: right;">63</td><td style = "text-align: right;">0</td><td style = "text-align: right;">4</td><td style = "text-align: right;">124</td><td style = "text-align: right;">197</td><td style = "text-align: right;">0</td><td style = "text-align: right;">0</td><td style = "text-align: right;">136</td><td style = "text-align: right;">1</td><td style = "text-align: right;">0.0</td><td style = "text-align: right;">2</td><td style = "text-align: left;">0</td><td style = "text-align: left;">3</td><td style = "text-align: right;">1</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">296</td><td style = "text-align: right;">41</td><td style = "text-align: right;">1</td><td style = "text-align: right;">2</td><td style = "text-align: right;">120</td><td style = "text-align: right;">157</td><td style = "text-align: right;">0</td><td style = "text-align: right;">0</td><td style = "text-align: right;">182</td><td style = "text-align: right;">0</td><td style = "text-align: right;">0.0</td><td style = "text-align: right;">1</td><td style = "text-align: left;">0</td><td style = "text-align: left;">3</td><td style = "text-align: right;">0</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">297</td><td style = "text-align: right;">59</td><td style = "text-align: right;">1</td><td style = "text-align: right;">4</td><td style = "text-align: right;">164</td><td style = "text-align: right;">176</td><td style = "text-align: right;">1</td><td style = "text-align: right;">2</td><td style = "text-align: right;">90</td><td style = "text-align: right;">0</td><td style = "text-align: right;">1.0</td><td style = "text-align: right;">2</td><td style = "text-align: left;">2</td><td style = "text-align: left;">6</td><td style = "text-align: right;">1</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">298</td><td style = "text-align: right;">57</td><td style = "text-align: right;">0</td><td style = "text-align: right;">4</td><td style = "text-align: right;">140</td><td style = "text-align: right;">241</td><td style = "text-align: right;">0</td><td style = "text-align: right;">0</td><td style = "text-align: right;">123</td><td style = "text-align: right;">1</td><td style = "text-align: right;">0.2</td><td style = "text-align: right;">2</td><td style = "text-align: left;">0</td><td style = "text-align: left;">7</td><td style = "text-align: right;">1</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">299</td><td style = "text-align: right;">45</td><td style = "text-align: right;">1</td><td style = "text-align: right;">1</td><td style = "text-align: right;">110</td><td style = "text-align: right;">264</td><td style = "text-align: right;">0</td><td style = "text-align: right;">0</td><td style = "text-align: right;">132</td><td style = "text-align: right;">0</td><td style = "text-align: right;">1.2</td><td style = "text-align: right;">2</td><td style = "text-align: left;">0</td><td style = "text-align: left;">7</td><td style = "text-align: right;">1</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">300</td><td style = "text-align: right;">68</td><td style = "text-align: right;">1</td><td style = "text-align: right;">4</td><td style = "text-align: right;">144</td><td style = "text-align: right;">193</td><td style = "text-align: right;">1</td><td style = "text-align: right;">0</td><td style = "text-align: right;">141</td><td style = "text-align: right;">0</td><td style = "text-align: right;">3.4</td><td style = "text-align: right;">2</td><td style = "text-align: left;">2</td><td style = "text-align: left;">7</td><td style = "text-align: right;">1</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">301</td><td style = "text-align: right;">57</td><td style = "text-align: right;">1</td><td style = "text-align: right;">4</td><td style = "text-align: right;">130</td><td style = "text-align: right;">131</td><td style = "text-align: right;">0</td><td style = "text-align: right;">0</td><td style = "text-align: right;">115</td><td style = "text-align: right;">1</td><td style = "text-align: right;">1.2</td><td style = "text-align: right;">2</td><td style = "text-align: left;">1</td><td style = "text-align: left;">7</td><td style = "text-align: right;">1</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">302</td><td style = "text-align: right;">57</td><td style = "text-align: right;">0</td><td style = "text-align: right;">2</td><td style = "text-align: right;">130</td><td style = "text-align: right;">236</td><td style = "text-align: right;">0</td><td style = "text-align: right;">2</td><td style = "text-align: right;">174</td><td style = "text-align: right;">0</td><td style = "text-align: right;">0.0</td><td style = "text-align: right;">2</td><td style = "text-align: left;">1</td><td style = "text-align: left;">3</td><td style = "text-align: right;">1</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">303</td><td style = "text-align: right;">38</td><td style = "text-align: right;">1</td><td style = "text-align: right;">3</td><td style = "text-align: right;">138</td><td style = "text-align: right;">175</td><td style = "text-align: right;">0</td><td style = "text-align: right;">0</td><td style = "text-align: right;">173</td><td style = "text-align: right;">0</td><td style = "text-align: right;">0.0</td><td style = "text-align: right;">1</td><td style = "text-align: left;">?</td><td style = "text-align: left;">3</td><td style = "text-align: right;">0</td></tr></tbody></table></div>



In the above data frame, the attributes are as follows:

| Variable Name | Role    | Type        | Demographic | Description                                           | Units | Missing Values |
|:-------------:|:-------:|:-----------:|:-----------:|:-----------------------------------------------------:|:-----:|:--------------:|
| age           | Feature | Integer     | Age         |                                                       | years | no             |
| sex           | Feature | Categorical | Sex         |                                                       |       | no             |
| cp            | Feature | Categorical |             |                                                       |       | no             |
| trestbps      | Feature | Integer     |             | resting blood pressure (on admission to the hospital) | mm Hg | no             |
| chol          | Feature | Integer     |             | serum cholestoral                                     | mg/dl | no             |
| fbs           | Feature | Categorical |             | fasting blood sugar > 120 mg/dl                       |       | no             |
| restecg       | Feature | Categorical |             |                                                       |       | no             |
| thalach       | Feature | Integer     |             | maximum heart rate achieved                           |       | no             |
| exang         | Feature | Categorical |             | exercise induced angina                               |       | no             |
| oldpeak       | Feature | Integer     |             | ST depression induced by exercise relative to rest    |       | no             |
          |


Complete attribute documentation:

    1. age: age in years
	2. sex: sex (1 = male; 0 = female)
	3. cp: chest pain type
		- Value 1: typical angina
		- Value 2: atypical angina
		- Value 3: non-anginal pain
		- Value 4: asymptomatic
	4. trestbps: resting blood pressure (in mm Hg on admission to the
	hospital)
	5. chol: serum cholestoral in mg/dl
	6.fbs: fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
	7. restecg: resting electrocardiographic results
		- Value 0: normal
		- Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
		- Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
	8. thalach: maximum heart rate achieved
	9. exang: exercise induced angina (1 = yes; 0 = no)
	10. oldpeak: ST depression induced by exercise relative to rest
	11. slope: the slope of the peak exercise ST segment
		- Value 1: upsloping
		- Value 2: flat
		- Value 3: downsloping
	12. ca: number of major vessels (0-3) colored by flourosopy (for calcification of vessels)
	13. thal: results of nuclear stress test (3 = normal; 6 = fixed defect; 7 = reversable defect)
	14. num: target variable representing diagnosis of heart disease (angiographic disease status) in any major vessel
		- Value 0: < 50% diameter narrowing
		- Value 1: > 50% diameter narrowing

## Data Interpretation

After collecting the data, it has been imported as a Data Frame. Now, to understand what we will do with this exercise, we need to analyze the data by means of Bayesian Logistic Regression.

With this type of analysis, we can make predictions on (typically) binary outcomes, based on a set of parameters. In this particular case, we are interested in predicting whether a patient will have heart disease based on a set of parameters such as age, chest pain, blood pressure, etc. 

In terms of the data available, we have a set of 303 observations (303 patients) whose symptoms and circumstances have been recorded, and the **outcome** is the heart disease diagnosis. To simplify things, this data set has a binary outcome, i.e. heart disease _present/not present_.

Additionally, this study is divided in two parts: first, I will set up the logistic regression model to include only one predictor, i.e., **age**. Afterwards, an analysis will be performed including two or more predictors.


```julia
# find the range for the age, to set the plot limits below
# min_age = minimum(df.age)
min_age = 15 
max_age = 85

# visualize data
p_data = scatter(df.age, df.num,
	legend = false,
	xlims = (min_age, max_age),
	color = :red,
	markersize = 5,
	title = "Probability of Heart Disease",
	xlabel = "Age (years)",
	ylabel = "Probability of Heart Disease",
    widen = true,
    dpi = 150
)

```


![svg](/images/20240109_Bayesian_Logistic_Regression/output_7_0.svg)


## Model Specification

In this stage of the workflow, we will specify the Bayesian model and then use `Turing.jl` to program it in Julia.

The model I will use for this analysis is a Bayesian Logistic Regression model, which relates the probability of heart disease to a _linear combination of the predictor variables_, using a logistic function. The model can be written as:

$$\begin{aligned}
y_i &\sim Bernoulli(p_i) \\\\
p_i &= \frac{1}{1+e^{-\eta_i}} \\\\
\eta_i &= \alpha + {\beta_1 x_{i,1}} + {\beta_2 x_{i,2}} + \ldots + {\beta_{13} x_{i,13}} \\\\
\alpha &\sim \mathcal{N}(\mu_\alpha,\sigma_\sigma) \\\\
\beta_j &\sim \mathcal{N}(\mu_{\beta},\sigma_{\beta}) \\\\
\end{aligned}$$

where $y_i$ is the outcome for the _i-th_ patient, $p_i$ is the probability of heart disease for the _i-th_ patient, $\eta_i$ is the linear predictor for the _i-th_ patient, $\alpha$ and $\beta_j$ are the intercept and coefficient for the _j-th_ predictor variable, respectively, and $x_{ij}$ is the value of the _j-th_ predictor variable for the _i-th_ patient.

The assumptions that I am making are:

1. The outcome variable follows a Bernoulli distribution, i.e. $y_i \sim Bernoulli(p_i)$, which is appropriate for binary outcomes
2. The predictor variables are linearly related to the log-odds of the outcome variable, i.e. $\log(\frac{p}{1-p})$ which is a common assumption for logistic regression models
3. The prior distributions for the model parameters are uniform, which are weakly informative and reflect my prior beliefs about the plausible range of the parameters

Regarding point (2):

That statement means that the log-odds of the outcome variable (the log of the odds ratio) can be expressed as a linear function of the predictor variables. Mathematically, this can be written as:

$$\log(\frac{p}{1-p}) = \alpha + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_k x_k$$

where $p$ is the probability of the outcome variable being 1, $x_1, x_2, \ldots, x_k$ are the predictor variables, and $\alpha, \beta_1, \beta_2, \ldots, \beta_k$ are the coefficients (parameters).

This assumption implies that the effect of each predictor variable on the log-odds of the outcome variable is contant, regardless of the values of the other predictor variables. It also implies that the relationship between the predictor variables and the probability of the outcome variable is non-linear, as the probability is obtained by applying the inverse of the log-odds function, which is the logistic function:

$$p = \frac{1}{1+e^{-(\alpha + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_k x_k)}}$$

The logistic function is an S-shaped curve that maps any real number to a value between 0 and 1. It has the property that as the linear predictor increases, the probability approaches 1, and as the linear predictor decreases, the probability approaches 0.


### Model Specification Using `Turing.jl`


```julia
# define the Bayesian model

@model function logit_model(predictors, disease)

	# priors
	α ~ Normal(0.0,10.0)
	β ~ Normal(0.0,10.0)

	# likelihood

	η = α .+ β.*predictors
	p = 1 ./ (1 .+ exp.(-η))     # remember to include the "."!
	for i in eachindex(p)
		disease[i] ~ Bernoulli(p[i])
	end
end
```




    logit_model (generic function with 2 methods)



#### Crank up the Bayes!

Run the model using `sample(model, sampler, opt_argument, samples, chains)`


```julia
# infer posterior probability

model = logit_model(df.age, df.num)
sampler = NUTS()
samples = 1_000
num_chains = 8 		# set the number of chains
chain = sample(model, sampler, MCMCThreads(), samples, num_chains)
```

    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39mFound initial step size
    [36m[1m└ [22m[39m  ϵ = 0.025
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39mFound initial step size
    [36m[1m└ [22m[39m  ϵ = 0.0125
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39mFound initial step size
    [36m[1m└ [22m[39m  ϵ = 0.025
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39mFound initial step size
    [36m[1m└ [22m[39m  ϵ = 0.0125
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39mFound initial step size
    [36m[1m└ [22m[39m  ϵ = 0.025
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39mFound initial step size
    [36m[1m└ [22m[39m  ϵ = 0.05
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39mFound initial step size
    [36m[1m└ [22m[39m  ϵ = 0.025
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39mFound initial step size
    [36m[1m└ [22m[39m  ϵ = 0.025
    [32mSampling (8 threads): 100%|█████████████████████████████| Time: 0:00:01[39m
    




    Chains MCMC chain (1000×14×8 Array{Float64, 3}):
    
    Iterations        = 501:1:1500
    Number of chains  = 8
    Samples per chain = 1000
    Wall duration     = 13.18 seconds
    Compute duration  = 100.1 seconds
    parameters        = α, β
    internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size
    
    Summary Statistics
     [1m parameters [0m [1m    mean [0m [1m     std [0m [1m    mcse [0m [1m  ess_bulk [0m [1m  ess_tail [0m [1m    rhat [0m [1m[0m ⋯
     [90m     Symbol [0m [90m Float64 [0m [90m Float64 [0m [90m Float64 [0m [90m   Float64 [0m [90m   Float64 [0m [90m Float64 [0m [90m[0m ⋯
    
               α   -3.0326    0.7453    0.0210   1242.6057   1246.3034    1.0043   ⋯
               β    0.0524    0.0134    0.0004   1224.4182   1259.7727    1.0037   ⋯
    [36m                                                                1 column omitted[0m
    
    Quantiles
     [1m parameters [0m [1m    2.5% [0m [1m   25.0% [0m [1m   50.0% [0m [1m   75.0% [0m [1m   97.5% [0m
     [90m     Symbol [0m [90m Float64 [0m [90m Float64 [0m [90m Float64 [0m [90m Float64 [0m [90m Float64 [0m
    
               α   -4.4814   -3.5432   -3.0127   -2.5183   -1.5973
               β    0.0264    0.0432    0.0521    0.0617    0.0789
    



**NOTE**: The above routine employs the `MCMCThreads()` method to sample multiple chains. However, to implement this, one needs to change the environment variables for the number of threads Julia can use. These two discussions might shed some light as to how to achieve this:
1. [https://docs.julialang.org/en/v1/manual/multi-threading/#man-multithreading](https://docs.julialang.org/en/v1/manual/multi-threading/#man-multithreading)
2. [https://discourse.julialang.org/t/julia-num-threads-in-vs-code-windows-10-wsl/28794](https://discourse.julialang.org/t/julia-num-threads-in-vs-code-windows-10-wsl/28794)

Of course, if you don't want to bother, then just change the last two functional lines in the cell above so that they read:

	# set number of chains - comment this out:
	# num_chains = 8
	
	# crank up the Bayes! - delete MCMCThreads() and num_chains
	chain = sample(model, sampler, samples)


#### Plot the MCMC Diagnostics


```julia
plot(chain, dpi = 150)
```
![png](/images/20240109_Bayesian_Logistic_Regression/output_15_0.svg)

#### Get the Summary Statistics


```julia
summarystats(chain)
```




    Summary Statistics
     [1m parameters [0m [1m    mean [0m [1m     std [0m [1m    mcse [0m [1m  ess_bulk [0m [1m  ess_tail [0m [1m    rhat [0m [1m[0m ⋯
     [90m     Symbol [0m [90m Float64 [0m [90m Float64 [0m [90m Float64 [0m [90m   Float64 [0m [90m   Float64 [0m [90m Float64 [0m [90m[0m ⋯
    
               α   -3.0326    0.7453    0.0210   1242.6057   1246.3034    1.0043   ⋯
               β    0.0524    0.0134    0.0004   1224.4182   1259.7727    1.0037   ⋯
    [36m                                                                1 column omitted[0m
    



### Plot and Interpret the Results

Ok, how do we interpret the results from a Bayesian approach? Let's start by plotting the results. This will help us understand not only the results, but really grasp the power of a Bayesian model in action.

From a frequentist or a machine learning approach, we would expect to find a function that models the data the best possible way, i.e. fit a model. If we were to visualize it, we would see one single sigmoid curve trying its best to explain the data.

How about this chart here, though? This chart is a collection of possible outcomes given that the _parameters_ $\alpha$ and $\beta$ in this case, are modeled as random variables with some probability distribution. Therefore, there is an uncertainty associated with them. This uncertainty is naturally _propagated_ onto the sigmoid function. Therefore, there is also an uncertainty associated with that sigmoid curve that we are trying to model. 

Again, below we can see a collection of possible outcomes given the parameter sample space. There is a darker region where most sigmoid functions turned out, and these tend to be the most probable sigmoid functions, or, in other words, these sigmoid functions are the most probable functions that could fit the data, considering the distributions of the parameters too!


```julia
Int(samples/10)
```




    100




```julia
x_line = 15:1:max_age

for i in 1:samples
    b = chain[i, 1, 1]
	m = chain[i, 2, 1]
	line(x) = m*x +b

	p(x) = 1 / (1 + exp(-line(x)) )

	plot!(p_data, x_line, p,
    	legend = false,
		linewidth = 2, color= :blue, alpha = 0.02, dpi = 150
	)
end

p_data
```

![png](/images/20240109_Bayesian_Logistic_Regression/output_20_0.svg)

### Making Predictions

So why go through all this trouble, you might be asking. Well, one of the reasons we want to use probabilistic models is, first, to make predictions. But I would go further than that: these models are useful when making informed decisions. Let's try this out.

Let's make predictions for different arbitrary ages (50, 60, 70, 80, 20):


```julia
new_Age = [50, 60, 70, 80, 20]
p_disease = fill(missing, length(new_Age))
predictions = predict(logit_model(new_Age, p_disease), chain)
summarystats(predictions)
```




    Summary Statistics
     [1m parameters [0m [1m    mean [0m [1m     std [0m [1m    mcse [0m [1m  ess_bulk [0m [1m ess_tail [0m [1m    rhat [0m [1m [0m ⋯
     [90m     Symbol [0m [90m Float64 [0m [90m Float64 [0m [90m Float64 [0m [90m   Float64 [0m [90m  Float64 [0m [90m Float64 [0m [90m [0m ⋯
    
      disease[1]    0.3855    0.4867    0.0055   7711.3762        NaN    0.9998    ⋯
      disease[2]    0.5258    0.4994    0.0055   8284.1301        NaN    0.9998    ⋯
      disease[3]    0.6432    0.4791    0.0056   7441.4457        NaN    1.0002    ⋯
      disease[4]    0.7555    0.4298    0.0050   7352.4368        NaN    0.9998    ⋯
      disease[5]    0.1224    0.3277    0.0039   7016.6404        NaN    1.0004    ⋯
    [36m                                                                1 column omitted[0m
    



#### Interpreting the predictions

The last operations make predictions of heart diseased based _on age only_. What the predictions mean is that, given the data, the probability distribution of an individual of age 50 to have heart disease has a mean of 0.379, and a standard deviation of 0.485 (this is highly uncertain, by the way).

Similarly, a 20-year-old individual has a probability with a mean of 0.13 and standard deviation of 0.336 of having heart disease.

These statistics are extremely powerful when you are trying to make decisions, such as when diagnosing Heart Disease. It stands to reason that, if you were a physician, you want to know what your model says might be wrong (or not) with your patient, but you also want to know how much you can trust that prediction. 

If your model classifies Patient X as having heart disease, you would probably want to know how sure you are of this. And this certainty comes partially from... you guessed it: your priors _and_ the data.

In the plot below, we can see the where the predictions lie. Note that these probabilities are on a continuum given by the sigmoid function. But we want our final decision to be a yes or a no. To do that, we need to set a decision threshold.

We will do that at the end of the next section.


```julia
for i in 1:length(new_Age)
    pred_mean = mean(predictions[:, i, 1])
    pred_plot = scatter!(p_data, (new_Age[i], pred_mean), dpi=150)
end

p_data
```
![png](/images/20240109_Bayesian_Logistic_Regression/output_24_0.svg)

## Model Specification Using Multiple Predictors

### Some Data Cleaning

In this part, I am using the `Turing.jl` documentation tutorial found in [https://turinglang.org/dev/tutorials/02-logistic-regression/](https://turinglang.org/dev/tutorials/02-logistic-regression/).

In the tutorial, they quite rightly incorporate a train/test split, and data normalization, which is the recommended practice. I didn't do it in the first part of this tutorial to keep things simple!

Here is how they handle the split and the data normalization using `MLUtils`.

	function split_data(df, target; at=0.70)
	    shuffled = shuffleobs(df)
	    return trainset, testset = stratifiedobs(row -> row[target], shuffled; p=at)
	end
	
	features = [:StudentNum, :Balance, :Income]
	numerics = [:Balance, :Income]
	target = :DefaultNum
	
	trainset, testset = split_data(data, target; at=0.05)
	for feature in numerics
	    μ, σ = rescale!(trainset[!, feature]; obsdim=1)
	    rescale!(testset[!, feature], μ, σ; obsdim=1)
	end
	
	# Turing requires data in matrix form, not dataframe
	train = Matrix(trainset[:, features])
	test = Matrix(testset[:, features])
	train_label = trainset[:, target]
	test_label = testset[:, target];


```julia
using MLDataUtils: shuffleobs, stratifiedobs, rescale!
using StatsFuns # we introduce this package so we can later call the 
                # logistic function directly instead of defining it manually as before
```


```julia
function split_data(df, target; at=0.70)
	shuffled = shuffleobs(df)
	return trainset, testset = stratifiedobs(row -> row[target], shuffled; p=at)
end
	
features = [:age, :cp, :chol]
target = :num
	
trainset, testset = split_data(df, target;)

# convert the feature columns to float64 to ensure compatibility with rescale!
for feature in features
    df[!, feature] = float.(df[!, feature])
end
	
for feature in features
    μ, σ = rescale!(trainset[!, feature]; obsdim=1)
    rescale!(testset[!, feature], μ, σ; obsdim=1)
end
	
# Turing requires data in matrix form, not dataframe
train = Matrix(trainset[!, features])
test = Matrix(testset[!, features])
train_label = trainset[!, target]
test_label = testset[!, target];
```

### Inference

Now that our data is formatted, we can perform our Bayesian logistic regression with multiple predictors: using chest pain (cp), age (age), resting bloodpressure (tresttbps) and cholesterol (chol) levels.



```julia
@model function logreg_multi(X, y)

	# priors
	intercept ~ Normal(0.0, 10.0)
	age ~ Normal(0.0, 10.0)
	cp ~ Normal(0.0, 10.0)
	chol ~ Normal(0.0, 10.0)

	n, _ = size(X)
	
	for i in 1:n
		# call the logistic function directly, instead of manually
		v = logistic(intercept + age*X[i,1] + cp*X[i,2] + chol*X[i,3])
		y[i] ~ Bernoulli(v)
	end

end
```




    logreg_multi (generic function with 2 methods)




```julia
X = train
y = train_label

println(size(train), size(test))
```

    (212, 3)(91, 3)
    

Now we build the model and create the chain:


```julia
model_multi = logreg_multi(X, y)
chain_multi = sample(model_multi, NUTS(), MCMCThreads(), 2_000, 8) # select 2000 samples directly
```

    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39mFound initial step size
    [36m[1m└ [22m[39m  ϵ = 1.6
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39mFound initial step size
    [36m[1m└ [22m[39m  ϵ = 0.8
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39mFound initial step size
    [36m[1m└ [22m[39m  ϵ = 0.8
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39mFound initial step size
    [36m[1m└ [22m[39m  ϵ = 0.8
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39mFound initial step size
    [36m[1m└ [22m[39m  ϵ = 0.8
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39mFound initial step size
    [36m[1m└ [22m[39m  ϵ = 1.6
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39mFound initial step size
    [36m[1m└ [22m[39m  ϵ = 0.8
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39mFound initial step size
    [36m[1m└ [22m[39m  ϵ = 1.6
    




    Chains MCMC chain (2000×16×8 Array{Float64, 3}):
    
    Iterations        = 1001:1:3000
    Number of chains  = 8
    Samples per chain = 2000
    Wall duration     = 11.32 seconds
    Compute duration  = 87.27 seconds
    parameters        = intercept, age, cp, chol
    internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size
    
    Summary Statistics
     [1m parameters [0m [1m    mean [0m [1m     std [0m [1m    mcse [0m [1m   ess_bulk [0m [1m   ess_tail [0m [1m    rhat[0m ⋯
     [90m     Symbol [0m [90m Float64 [0m [90m Float64 [0m [90m Float64 [0m [90m    Float64 [0m [90m    Float64 [0m [90m Float64[0m ⋯
    
       intercept   -0.2821    0.1647    0.0012   20113.9419   13042.8456    1.0003 ⋯
             age    0.6003    0.1760    0.0013   18327.5449   12926.7418    1.0001 ⋯
              cp    1.0699    0.1922    0.0014   19583.1899   13534.2405    0.9999 ⋯
            chol   -0.0073    0.1641    0.0012   18280.4944   12242.8280    1.0004 ⋯
    [36m                                                                1 column omitted[0m
    
    Quantiles
     [1m parameters [0m [1m    2.5% [0m [1m   25.0% [0m [1m   50.0% [0m [1m   75.0% [0m [1m   97.5% [0m
     [90m     Symbol [0m [90m Float64 [0m [90m Float64 [0m [90m Float64 [0m [90m Float64 [0m [90m Float64 [0m
    
       intercept   -0.6118   -0.3923   -0.2817   -0.1711    0.0388
             age    0.2645    0.4792    0.5964    0.7178    0.9575
              cp    0.7106    0.9372    1.0636    1.1963    1.4603
            chol   -0.3283   -0.1177   -0.0080    0.1025    0.3151
    



### Plot the MCMC Diagnostics


```julia
plot(chain_multi, dpi=150)
```

![png](/images/20240109_Bayesian_Logistic_Regression/output_34_0.svg)

### Summary Statistics


```julia
summarystats(chain_multi)
```




    Summary Statistics
     [1m parameters [0m [1m    mean [0m [1m     std [0m [1m    mcse [0m [1m   ess_bulk [0m [1m   ess_tail [0m [1m    rhat[0m ⋯
     [90m     Symbol [0m [90m Float64 [0m [90m Float64 [0m [90m Float64 [0m [90m    Float64 [0m [90m    Float64 [0m [90m Float64[0m ⋯
    
       intercept   -0.2821    0.1647    0.0012   20113.9419   13042.8456    1.0003 ⋯
             age    0.6003    0.1760    0.0013   18327.5449   12926.7418    1.0001 ⋯
              cp    1.0699    0.1922    0.0014   19583.1899   13534.2405    0.9999 ⋯
            chol   -0.0073    0.1641    0.0012   18280.4944   12242.8280    1.0004 ⋯
    [36m                                                                1 column omitted[0m
    



## Thank you!

And that concludes this little tutorial showcasing the power of a Bayesian model and the fun of using Julia. Thank you for stopping by!

Victor Flores

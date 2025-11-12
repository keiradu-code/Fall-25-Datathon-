# Fall-25-Datathon-

Summary of Business Problem Steps and Our Plan! 

DO MODEL LIKE NORMAL, EXPLAIN RELATIVE TO BUSINESS DESCRIPTION!!!!

Notes: in the model data, "sample" column differentiates training and testing data. "1|bld" == training/building data, "2|val" == testing data. 

MODELING
Develop a ML model to predict claim cost per policy term ('claimcst0'). What does that mean??
    Predict how much money the insurance company expects to pay out in claims !FOR A GIVEN CUSTOMER! during their policy term. 

Start with model design --> data collection/cleaning --> data exploration --> model selection --> model implementation

1) Variable reduction: suggested method is 'VarClusHi'; a python package. 

2) Tree-based Models and Hyperparameter Tuning (Primary modeling technique): suggested interactive hyperparameter tuning approach (tuning 2-3 at a time). 

3)Compare Tree-Based Perfomance against "TabPFN" (transformer)

4) Frequency-Severity Modeling & Exposure Handling: estimating claim cost per policy is complex b.c. of 2 challenges: 
    1. Total claim cost is a product of frequency and severity. 
    1a. So build seperate models for frequency and severity, then combine their predictions
    2. Policy terms have varying exposure durations.
    2a. So account for variable exposures using weighted likelihood estimation or offset techniques

5) Use 'SHAP' to quantify the contribution of each feature to individual predictions as well as the overall model. 


Plan? 
1) Clean data (missing values, data types, check outliers)
2) encode categorical variables
2) EDA: distribution of variables, relationship between exposure and claim frequency/severity, summarisze which factors seem most associated with higher losses. 
3) Create claim_freq = numclaims/exposure
4) Create severity = claimcst0 / numclaims (for claims > 0; i.e. a claim occured)
5) variable reduction
6) split data into bld and val. 
7) claimcst0 = predictors. Train tree model
8) Evaluate tree model (exposure-weighted)
9) Improve results with targeted tuning. 
Not sure if TabFPN stuff is neccessary
10) Whatever SHAP is
11) Interpret model and prepare business insights. 


Breakdown: expected_claim_cost = frequency * severity
frequency = num of claims/ exposure
severity = claim cost / num of claims
What is claim frequency? 
    Number of claims that occur for a given exposure period (per policy term in this case)
What is claim severity? 
    Average cost per claim, given that a claim occured. 
What is claim exposure?
    Amount of time or risk preiod a policy is in force. 
    Ensures that when you compare customers, you're doing it fairly; someone insured for 6 months shouldn't have the same expected claim cost as someone ensured for a year. 

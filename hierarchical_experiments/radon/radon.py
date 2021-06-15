import numpy as np
import pandas as pd
import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt

import pystan

# original model code
varying_intercept = """
data {
    int<lower=0> J; # number of counties
    int<lower=0> N; # number of observations
    int<lower=1,upper=J> county[N]; # which county does each observation belong to?
    vector[N] x;
    vector[N] y;
} 
parameters {
    vector[J] a;
    real b;
    real mu_a;
    real<lower=0,upper=100> sigma_a;
    real<lower=0,upper=100> sigma_y;
} 
transformed parameters {

    vector[N] y_hat;

    for (i in 1:N)
      y_hat[i] <- a[county[i]] + x[i] * b;
}
model {
    sigma_a ~ uniform(0, 100);
    a ~ normal (mu_a, sigma_a);

    b ~ normal (0, 1);

    sigma_y ~ uniform(0, 100);
    y ~ normal(y_hat, sigma_y);
    }
"""

# weighted model code
# different prior over a than original code, so that overall
# prior is proper
weighted_varying_intercept = """
data {
    int<lower=0> J; 
    int<lower=0> N; 
    int<lower=1,upper=J> county[N];
    vector[N] x;
    vector[N] y;
    vector[N] w; 
} 

parameters {
    vector[J] a;
    real b;
    real mu_a;
    real<lower=0,upper=100> sigma_a;
    real<lower=0,upper=100> sigma_y;
} 

transformed parameters {
    vector[N] y_hat;

    for (i in 1:N)
      y_hat[i] <- a[county[i]] + x[i] * b;
}

model {
    mu_a ~ normal(0, 100);
    sigma_a ~ uniform(0, 100);
    b ~ normal (0, 1);
    sigma_y ~ uniform(0, 100);
    
    a ~ normal (mu_a, sigma_a);

    for (i in 1:N)
      // target += w[i]*(-square(y[i]-y_hat[i])/(2*square(sigma_y))-log(sqrt(2*pi()*square(sigma_y))));
      target += w[i]*normal_lpdf(y[i] | y_hat[i], sigma_y);
}

generated quantities {
    vector[N] ll;
    
    for (i in 1:N)
        ll[i] = normal_lpdf(y[i] | y_hat[i], sigma_y);
    
}

"""

## model code for prior
prior_code = """
data {
    int<lower=0> J; 
    int<lower=0> N; 
    int<lower=1,upper=J> county[N];
    vector[N] x;
    vector[N] y;
    vector[N] w; 
} 

parameters {
    vector[J] a;
    real b;
    real mu_a;
    real<lower=0,upper=100> sigma_a;
    real<lower=0,upper=100> sigma_y;
} 

transformed parameters {
    vector[N] y_hat;

    for (i in 1:N)
      y_hat[i] <- a[county[i]] + x[i] * b;
}

model {
    mu_a ~ normal(0, 100);
    sigma_a ~ uniform(0, 100);
    b ~ normal (0, 1);
    sigma_y ~ uniform(0, 100);
    
    a ~ normal (mu_a, sigma_a);
}

generated quantities {
    vector[N] ll;
    for (i in 1:N)
        ll[i] = normal_lpdf(y[i] | y_hat[i], sigma_y);
}
"""

def load_data():
    # load radon data
    srrs2 = pd.read_csv('../data/srrs2.dat')
    srrs2.columns = srrs2.columns.map(str.strip)
    srrs_mn = srrs2.assign(fips=srrs2.stfips*1000 + srrs2.cntyfips)[srrs2.state=='MN']

    cty = pd.read_csv('../data/cty.dat')
    cty_mn = cty[cty.st=='MN'].copy()
    cty_mn[ 'fips'] = 1000*cty_mn.stfips + cty_mn.ctfips

    srrs_mn = srrs_mn.merge(cty_mn[['fips', 'Uppm']], on='fips')
    srrs_mn = srrs_mn.drop_duplicates(subset='idnum')
    u = np.log(srrs_mn.Uppm)

    n = len(srrs_mn)
    n_county = srrs_mn.groupby('county')['idnum'].count()

    srrs_mn.county = srrs_mn.county.str.strip()
    mn_counties = srrs_mn.county.unique()
    counties = len(mn_counties)

    county_lookup = dict(zip(mn_counties, range(len(mn_counties))))
    county = srrs_mn['county_code'] = srrs_mn.county.replace(county_lookup).values
    radon = srrs_mn.activity
    srrs_mn['log_radon'] = log_radon = np.log(radon + 0.1).values
    floor_measure = srrs_mn.floor.values
    
    data = {'N': len(log_radon),
                      'J': len(n_county),
                      'county': county+1, # Stan % counts starting at 1
                      'x': floor_measure,
                      'w': np.ones(len(log_radon)),
                      'y': log_radon}
            
    return data
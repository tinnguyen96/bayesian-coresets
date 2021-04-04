import numpy as np
import pandas as pd
import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt

import pystan

if __name__ == "__main__":
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
    sigma_a ~ uniform(0, 100);
    a ~ normal (mu_a, sigma_a);

    b ~ normal (0, 1);

    sigma_y ~ uniform(0, 100);
    for (i in 1:N)
      target += w[i]*(-square(y[i]-y_hat[i])/(2*square(sigma_y))-log(sqrt(2*pi()*square(sigma_y))));
  }
  """

  # load radon data
  srrs2 = pd.read_csv('data/srrs2.dat')
  srrs2.columns = srrs2.columns.map(str.strip)
  srrs_mn = srrs2.assign(fips=srrs2.stfips*1000 + srrs2.cntyfips)[srrs2.state=='MN']

  cty = pd.read_csv('data/cty.dat')
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

  # varying intercept fit
  print("Normal model\n")
  varying_intercept_data = {'N': len(log_radon),
                          'J': len(n_county),
                          'county': county+1, # Stan % counts starting at 1
                          'x': floor_measure,
                          'y': log_radon}

  path = 'stan_cache/normal_model.pkl'
  if os.path.isfile(path):
      sm = pickle.load(open(path, 'rb'))
  else:
      sm = pystan.StanModel(model_code=varying_intercept)
      with open(path, 'wb') as f: pickle.dump(sm, f)

  varying_intercept_fit = sm.sampling(data=varying_intercept_data, iter=1000, chains=2)

  a_sample = pd.DataFrame(varying_intercept_fit['a'])

  sns.set(style="ticks", palette="muted", color_codes=True)

  # Plot the orbital period with horizontal boxes
  plt.figure(figsize=(16, 6))
  sns.boxplot(data=a_sample, whis=np.inf, color="c")

  # weighted varying intercept fit
  print("Weighted LL model\n")
  print("weighted but not missing anything")
  w = np.ones(len(log_radon))
  # print("w is ", w)

  w_varying_intercept_data = {'N': len(log_radon),
                          'J': len(n_county),
                          'county': county+1, # Stan counts starting at 1,
                          'w':w,
                          'x': floor_measure,
                          'y': log_radon}

  path = 'stan_cache/weighted_model_nomissing.pkl'
  if os.path.isfile(path):
      sm = pickle.load(open(path, 'rb'))
  else:
      sm = pystan.StanModel(model_code=weighted_varying_intercept)
      with open(path, 'wb') as f: pickle.dump(sm, f)

  w_varying_intercept_fit = sm.sampling(data=w_varying_intercept_data, iter=1000, chains=2)

  a_sample = pd.DataFrame(w_varying_intercept_fit['a'])

  sns.set(style="ticks", palette="muted", color_codes=True)

  # Plot the orbital period with horizontal boxes
  plt.figure(figsize=(16, 6))
  sns.boxplot(data=a_sample, whis=np.inf, color="c")

  np.random.seed(0)
  missing_coord = np.random.choice(np.arange(len(log_radon)))
  print("Leaving component = %d" %missing_coord)
  w[missing_coord] = 0 

  w_varying_intercept_data = {'N': len(log_radon),
                          'J': len(n_county),
                          'county': county+1, # Stan counts starting at 1,
                          'w':w,
                          'x': floor_measure,
                          'y': log_radon}

  path = 'stan_cache/weighted_model_missing=%d.pkl' %missing_coord
  if os.path.isfile(path):
      sm = pickle.load(open(path, 'rb'))
  else:
      sm = pystan.StanModel(model_code=weighted_varying_intercept)
      with open(path, 'wb') as f: pickle.dump(sm, f)

  w_varying_intercept_fit = sm.sampling(data=w_varying_intercept_data, iter=1000, chains=2)

  a_sample = pd.DataFrame(w_varying_intercept_fit['a'])

  sns.set(style="ticks", palette="muted", color_codes=True)

  # Plot the orbital period with horizontal boxes
  plt.figure(figsize=(16, 6))
  sns.boxplot(data=a_sample, whis=np.inf, color="c")
from __future__ import print_function
import numpy as np
import bayesiancoresets as bc
from scipy.optimize import minimize
from scipy.linalg import solve_triangular
import time
import sys, os
import argparse
import pystan
import pickle

#make it so we can import models/etc from parent folder
sys.path.insert(1, os.path.join(sys.path[0], '../../examples/common'))
import mcmc
import results
import plotting
import radon

parser = argparse.ArgumentParser(description="Runs Hilbert coreset construction on a model and dataset")

parser.add_argument("--proj_dim", type=int, default=500, help="The number of samples taken when discretizing log likelihoods for these experiments")

parser.add_argument('--coreset_size_max', type=int, default=1000, help="The maximum coreset size to evaluate")
parser.add_argument('--coreset_num_sizes', type=int, default=7, help="The number of coreset sizes to evaluate")
parser.add_argument('--coreset_size_spacing', type=str, choices=['log', 'linear'], default='log', help="The spacing of coreset sizes to test")
parser.add_argument('--opt_itrs', type=str, default = 100, help="Number of optimization iterations (for methods that use iterative weight refinement)")
parser.add_argument('--step_sched', type=str, default = "lambda i : 1./(1+i)", help="Optimization step schedule (for methods that use iterative weight refinement); entered as a python lambda expression surrounded by quotes")
parser.add_argument('--trial', type=int, help="The trial number - used to initialize random number generation (for replicability)")

arguments = parser.parse_args()

# load data
data_dict_, prior_dict_ = radon.load_data()

stan_representation = radon.weighted_varying_intercept

# load stanfit (or make one if no cache exists)
path_with_data = 'stancache/weighted_radon.pkl'
if os.path.isfile(path_with_data):
    sm = pickle.load(open(path_with_data, 'rb'))
else:
    sm = pystan.StanModel(model_code=stan_representation)
    with open(path_with_data, 'wb') as f: pickle.dump(sm, f)
        
path_without_data = 'stancache/radon_prior.pkl'
if os.path.isfile(path_without_data):
    sm_prior = pickle.load(open(path_without_data, 'rb'))
else:
    sm_prior = pystan.StanModel(model_code=radon.prior_code)
    with open(path_without_data, 'wb') as f: pickle.dump(sm_prior, f)

## load pystan fit object, which has functions to eval log_likelihood
## we don't really need the samples from this stanfit, but it's convenient
## to automatically evaluate log_likelihood (as opposed to making a separate
## function to do it)

print('Creating likelihood evaluator')
stanfit = sm.sampling(data=data_dict_)
# print(stanfit)
likelihood = lambda pts, th: stanfit.log_prob(stanfit.unconstrain_pars(th))

# create projectors
print('Creating black box projector')

def weighted_sampler(n, wts, pts):
    if (wts is None) or (len(wts) == 0):
        priorfit = sm_prior.sampling(data=prior_dict_, iter=2*n, chains=1)
        samples = priorfit.extract()
    else:
        weighted_data = data_dict_.copy()
        weighted_data["w"] = wts
        smallfit = sm.sampling(data=weighted_data, iter=2*n, chains=1)
        samples = smallfit.extract()
    return samples

prj_bb = bc.BlackBoxProjector(weighted_sampler, arguments.proj_dim, likelihood, None)

## we actually don't need to pass in data if we don't use n_subsample_select or 
## n_subsample_opt
sparsevi = bc.SparseVICoreset(None, prj_bb, opt_itrs = arguments.opt_itrs, step_sched = eval(arguments.step_sched))

Ms = np.unique(np.linspace(1, 100, 4, dtype=np.int32))

alg = sparsevi

for m in range(Ms.shape[0]):
    # print('M = ' + str(Ms[m]) + ': coreset construction, '+ arguments.alg + ' ' + arguments.dataset + ' ' + str(arguments.trial))
    #this runs alg up to a level of M; on the next iteration, it will continue from where it left off
    t0 = time.process_time()
    itrs = (Ms[m] if m == 0 else Ms[m] - Ms[m-1])
    alg.build(itrs)
    t_alg += time.process_time()-t0
    wts, pts, idcs = alg.get()

    print('M = ' + str(Ms[m]) + ': MCMC')
    # Use MCMC on the coreset, measure time taken 
    weighted_data = data_dict_.copy()
    weighted_data["w"] = wts
    coresetfit = sm.sampling(data=weighted_data, iter=2*1000, chains=1)
    print(coresetfit)
    samples = coresetfit.extract()
    print()
    
print("Completed\n")
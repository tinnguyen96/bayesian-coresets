"""
TODO 
"""

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
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

#make it so we can import models/etc from parent folder
sys.path.insert(1, os.path.join(sys.path[0], '../../examples/common'))
import results
import plotting
import radon
import names_and_parser

if __name__ == "__main__":

    arguments = names_and_parser.parse_args()
    print(arguments)
    
    savedir = names_and_parser.make_radon_name(arguments)
    if not os.path.isdir(savedir):
        print("Making dir at %s" %savedir)
        os.makedirs(savedir)

    np.random.seed(arguments.trial)
    bc.util.set_verbosity(arguments.verbosity) 

    # load data
    data_dict_ = radon.load_data()
    N = data_dict_["N"]
    print("Number of observations is %d" %N)

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

    # create projectors
    print('Creating stan fit projector')
    ll_names = ["ll[%s]" %idx for idx in range(1,N+1)]
    num_weighted_sampler = 0 # use to set seed for stanmodel sampling in weighted_sampler
    
    def weighted_sampler(n, wts, pts):
        """
        Inputs:
            n: scalar, number of samples to draw
            wts: (m,) array, coreset weights where m is current number of coreset
            pts: (m, D+1) array, coreset data, D is original number of dimensions
                first column of pts is the index in the original data matrix

        Output:
            full_ll: (N, n) array of log likelihood, where N is original number 
                of observations
            wts_ll: (m, n) array of log likelihood for coreset points
        """
        print("wts", wts)
        print("pts", pts)
        global num_weighted_sampler
        print("num_weighted_sampler", num_weighted_sampler)
        if wts is None or pts is None or pts.shape[0] == 0:
            priorfit = sm_prior.sampling(data=data_dict_, iter=2*n, chains=1, 
                                         refresh=0, seed=num_weighted_sampler)
            df = priorfit.to_dataframe()
        else:
            weighted_data = data_dict_.copy()
            wts_for_stan = np.zeros(N)
            for idx_in_wts, weight in enumerate(wts):
                idx_in_data = int(pts[0,idx_in_wts])
                wts_for_stan[idx_in_data] = weight
            weighted_data["w"] = wts_for_stan
            smallfit = sm.sampling(data=weighted_data, iter=2*n, chains=1, 
                                   refresh=0, seed=num_weighted_sampler)
            df = smallfit.to_dataframe()
        full_ll = df[ll_names]
        full_ll = np.array(full_ll).transpose()

        wts_ll = np.zeros((len(wts), n))
        for idx_in_wts, weight in enumerate(wts):
            idx_in_data = int(pts[0,idx_in_wts])
            wts_ll[idx_in_wts, :] = weight*full_ll[idx_in_data,:]
        num_weighted_sampler += 1
        return full_ll, wts_ll

    prj_sf = bc.StanFitProjector(weighted_sampler, N, arguments.proj_dim)

    ## we actually don't need to pass in data if we don't use n_subsample_select or 
    ## n_subsample_opt
    data_for_svi = np.vstack((np.arange(N), data_dict_["x"], data_dict_["y"])).transpose()
    # print("data_for_svi.shape", data_for_svi.shape)

    sparsevi = bc.SparseVICoreset(data=data_for_svi, ll_projector=prj_sf, 
                                  n_subsample_select=None, n_subsample_opt=None, 
                                  opt_itrs = arguments.opt_itrs, step_sched = eval(arguments.step_sched))

    Ms = np.unique(np.linspace(1, 100, 4, dtype=np.int32))

    alg = sparsevi

    print("Initiating coreset construction")
    t_alg = 0
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

        wts_for_stan = np.zeros(N)
        for idx_in_wts, weight in enumerate(wts):
            idx_in_data = int(pts[0,idx_in_wts])
            wts_for_stan[idx_in_data] = weight
        weighted_data["w"] = wts_for_stan
        coresetfit = sm.sampling(data=weighted_data, iter=2*1000, chains=4, seed=arguments.trial)
        
        savepath = savedir + "/coreset_M=%d.pkl" %Ms[m]
        print("Will save coreset fit to %s" %savepath)
        with open(savepath, 'wb') as f: pickle.dump(coresetfit, f)
            
        a_sample = pd.DataFrame(coresetfit['a'])
        sns.set(style="ticks", palette="muted", color_codes=True)
        # Plot the orbital period with horizontal boxes
        plt.figure(figsize=(16, 6))
        sns.boxplot(data=a_sample, whis=np.inf, color="c")
        savepath = savedir + "/coreset_M=%d_var=a.png" %Ms[m]
        plt.title("coreset_M=%d_var=a" %Ms[m])
        plt.savefig(savepath)
        print()

    print("Completed\n")
"""
Define functions that read in command line arguments and functions
that return names of directories.
"""

import argparse
import os 
import re
import fnmatch

## ------------------------------------------------------------------
## Argument parsers

def parse_args():
    parser = argparse.ArgumentParser()
    parser.set_defaults(est_type="estimates", data_dir = "../data/",
            results_dir="../estimation_results/", data_root="gene",t=5, num_prop=5, num_scans=5,
            Ndata=200, D=50, max_iter=1000, max_time=300, sd=5.0, sd0=5.0,pool_size=10, n_replicates=10,
            alpha=0.5, k=10, m=20, joint_init_type = "swap", init_type="all_same", pi0_its=0, Gibbs_coupling="Optimal")
    
    ## DP model hyper-params
    parser.add_argument("--sd", type=float, dest="sd",
                    help="std of observational likelihood")
    parser.add_argument("--sd0", type=float, dest="sd0",
                    help="std of prior distribution over cluster means")
    parser.add_argument("--alpha", type=float, dest="alpha",
                    help="concentration of Dirichlet parameter to generate cluster weights")
    parser.add_argument("--track_means", dest='track_means', action='store_true', help="do we instantiate cluster centers during sampling")
    
    ## time budget and number of replicates
    parser.add_argument("--pool_size", type=int, dest="pool_size",
            help="number of processes to run in parallel")
    parser.add_argument("--n_replicates", type=int, dest="n_replicates",
            help="number of times to repeat the experiment")
    parser.add_argument("--max_iter", type=int, dest="max_iter",
                    help="maximum number of sweeps through data when computing truth")
    parser.add_argument("--max_time", type=int, dest="max_time",
                    help="maximum processor time to run each replicate")
    parser.add_argument("--profile",  dest="profile", 
                        help="whether to profile the functions", action='store_true')
    
    ## information about data
    parser.add_argument("--data_dir", type=str, dest="data_dir",
                    help="root directory containing data files")
    parser.add_argument("--data_root", type=str, dest="data_root",
                    help="type of data (synthetic or gene expression data)")
    parser.add_argument("--results_dir", type=str, dest="results_dir",
                    help="where to save results")
    parser.add_argument("--Ndata", type=int, dest="Ndata",
                    help="number of observations")
    parser.add_argument("--D", type=int, dest="D",
                    help="number of features")
    
    ## settings for estimator construction
    parser.add_argument("--sampler_type", type=str, dest="sampler_type", choices=["Gibbs", "Jain_Neal"])
    parser.add_argument("--num_prop", type=int, dest="num_prop", 
                        help="number of split-merge proposals during each Jain-Neal iteration")
    parser.add_argument("--t", type=int, dest="t", 
                        help="number of restricted Gibbs scan to reach launch state")
    parser.add_argument("--num_scans", type=int, dest="num_scans", 
                        help="number of Gibbs scan after all split-merge proposals")
    parser.add_argument("--est_type", type=str, dest="est_type",
                    help="type of estimator", choices=["truth", "quad", "coupled", "single"])
    parser.add_argument("--h_type", type=str, dest="h_type",
                    help="type of function") 
    parser.add_argument("--coupled_corrections",  
                        dest='coupled_corrections', action='store_true')
    parser.add_argument("--save_memory", help="if true, only save ests, not states",
                        dest='save_memory', action='store_true')
    parser.add_argument("--Gibbs_coupling",type=str, dest='Gibbs_coupling', choices=["Optimal"])
    parser.add_argument("--JainNeal_coupling",type=str, dest='JainNeal_coupling', choices=["naive"])
    parser.add_argument("--early_coupling", help="if true, couple the cluster means sampling step",
                        dest='early_coupling', action='store_true')
    parser.add_argument("--metric", type=str, choices=["Hamming", "VI"], help="choice of metric between partitions",
                        dest='metric')
    parser.add_argument("--k", type=int, dest="k",
                    help="length of burn-in period")
    parser.add_argument("--m", type=int, dest="m",
                    help="minimum number of sweeps to perform when coupling")
    parser.add_argument("--init_type", type=str, dest="init_type",
                    help="how to initialize the Gibbs sampler")
    parser.add_argument("--pi0_its", type=int, dest="pi0_its",
                    help="how many sweeps to move from initialization type")
    parser.add_argument("--joint_init_type", type=str, dest="joint_init_type",
                    help="how to jointly initialize the four chains",
                    choices=['indep', 'swap'])

    # type of experiment to run
    parser.add_argument("--exp_type", type=str, dest="exp_type", choices=["estimation", "meeting", "runtime"],
                    help="type of experiment to do")
    
    ## options for compilation
    parser.add_argument("--compile_type", type=str, dest="compile_type",
                    help="type of compilation to do")

    options = parser.parse_args()
    return options
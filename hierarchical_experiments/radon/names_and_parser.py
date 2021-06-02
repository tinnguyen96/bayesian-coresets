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
    parser = argparse.ArgumentParser(description="Runs Hilbert coreset construction on a model and dataset")

    parser.add_argument("--proj_dim", type=int, default=500, 
                        help="The number of samples taken when discretizing log likelihoods for these experiments")
    parser.add_argument("--results_dir", type=str, default="results", 
                        help="folder in which to put results")
    parser.add_argument('--opt_itrs', type=int, default = 100, 
                        help="Number of optimization iterations (for methods that use iterative weight refinement)")
    parser.add_argument('--step_sched', type=str, default = "lambda i : 1./(1+i)", 
                        help="Optimization step schedule (for methods that use iterative weight refinement); entered as a python lambda expression surrounded by quotes")
    parser.add_argument('--trial', type=int, default=0,
                        help="The trial number - used to initialize random number generation (for replicability)")

    parser.add_argument('--verbosity', type=str, default="error", choices=['error', 'warning', 'critical', 'info', 'debug'], 
                        help="The verbosity level.")
    parser.add_argument('--save_samples', action='store_true', default=False)

    options = parser.parse_args()
    return options

def make_radon_name(options):
    savedir = "%s/opt_itrs=%d_trial=%d" %(options.results_dir, options.opt_itrs, options.trial)
    return savedir
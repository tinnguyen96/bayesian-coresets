#!/bin/bash

module load anaconda/2021a
source activate pystan

python -u main.py --opt_itrs 10 --trial 0
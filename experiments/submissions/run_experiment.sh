#!/bin/bash
#SBATCH --ntasks-per-node=4 # core count
#SBATCH -o ../logs/gaussian.sh.log-%j

module load anaconda/2020a 
cd ..

for alg in "US" "GIGA-REAL" "GIGA-REAL-EXACT" "GIGA-OPT" "GIGA-OPT-EXACT" "SVI-EXACT" "SVI"
do 
    for ID in {1..3}
    do

        python main.py --alg $alg --trial $ID run
    done
done

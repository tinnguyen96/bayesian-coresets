#!/bin/bash 

module load anaconda/2021a
cd ../../../examples/simple_lr

# for alg in "US" "GIGA-REAL" "GIGA-REAL-EXACT" "GIGA-OPT" "GIGA-OPT-EXACT" "SVI-EXACT" "SVI"
for alg in "US" "GIGA-REAL"
do 
    for ID in {1..3}
    do
        python main.py --alg $alg --trial $ID run
    done
done

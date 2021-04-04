#!/bin/bash  
#SBATCH --exclusive
#SBATCH -o ../logs/whatever.sh.log-%j

module load anaconda/2020a 
source activate py2_frag_topics
cd ..

nBatch=5
nLap=5
max_doc=100000

K=5
python -u run.py --max_doc $max_doc --K $K --nLap $nLap --nBatch $nBatch --plot_type top_words

K=50
python -u run.py --max_doc $max_doc --K $K --nLap $nLap --nBatch $nBatch --plot_type topic_proportions
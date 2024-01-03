#!/bin/bash
#SBATCH --job-name=multi
#SBATCH --partition=brtx6-ir
#SBATCH --nodelist=brtx603
#SBATCH --gpus=1
#SBATCH --mail-user=bchang26@jhu.edu
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END

# Move to correct directory
cd /brtx/604-nvme1/bchang26/multi/NLP-HW6/code

# See current directory
cd /path/to/your/directory
echo "Current directory (Main): $(pwd)"

# Function to submit a job
submit_job() {
    job_name=$1
    command=$2

    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=$job_name
#SBATCH --partition=brtx6-ir
#SBATCH --nodelist=brtx603
#SBATCH --gpus=1
#SBATCH --mail-user=bchang26@jhu.edu
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END

# Activate Conda environment
source /home/bchang26/anaconda3/etc/profile.d/conda.sh
conda activate nlp-class

# See current directory
cd /path/to/your/directory
echo "Current directory (Sub): $(pwd)"

echo "Running $job_name"
echo "/home/bchang26/anaconda3/envs/nlp-class/bin/python $2"
/home/bchang26/anaconda3/envs/nlp-class/bin/python $2
EOT
}

# submit_job "multi" "tag.py ../data/icdev -t ../data/icsup --save_path ic_hmm.pkl"
# submit_job "multi" "tag.py ../data/icdev -t ../data/{icsup,icraw} --save_path ic_hmm_raw.pkl"
# submit_job "multi" "tag.py ../data/endev -l ../data/words-50.txt -t ../data/ensup --save_path en_hmm.pkl"
# submit_job "multi" "tag.py ../data/endev -l ../data/words-50.txt -t ../data/{ensup,enraw} --save_path en_hmm_raw.pkl"
# submit_job "multi" "tag.py ../data/endev -l ../data/words-50.txt -t ../data/ensup --save_path en_hmm_awesome.pkl --awesome"
# submit_job "multi" "tag.py ../data/icdev -t ../data/icsup --crf --save_path ic_crf.pkl"
# submit_job "multi" "tag.py ../data/endev -l ../data/words-50.txt -t ../data/ensup --crf --save_path en_crf.pkl"
# submit_job "multi" "tag.py ../data/endev -l ../data/words-50.txt -t ../data/{ensup,enraw} --crf --save_path en_crf_raw.pkl"
submit_job "multi" "tag.py ../data/endev -l ../data/words-50.txt -t ../data/ensup --crf --save_path en_crf_birnn.pkl" # RUN THIS AFTER
# submit_job "multi" "tag.py ../data/endev -l ../data/words-50.txt -t ../data/ensup --save_path base.pkl"
# submit_job "multi" "tag.py ../data/endev -l ../data/words-50.txt -t ../data/ensup --awesome --save_path all.pkl"
# submit_job "multi" "tag.py ../data/endev -l ../data/words-50.txt -t ../data/ensup --awesome --save_path base_log_counts.pkl"
# submit_job "multi" "tag.py ../data/endev -l ../data/words-50.txt -t ../data/ensup --awesome --save_path base_affixes.pkl"
# submit_job "multi" "tag.py ../data/endev -l ../data/words-50.txt -t ../data/ensup --awesome --save_path base_word_couts.pkl"


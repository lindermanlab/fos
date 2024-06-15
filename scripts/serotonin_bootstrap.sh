#!/bin/bash
#SBATCH --job-name=serotonin-fos-bootstrap
#SBATCH --nodes=1 --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=20gb
#SBATCH --time=23:59:00
#SBATCH -p swl1

echo ${SLURM_ARRAY_TASK_ID}
#source ~/bin/load_modules.sh
cd /home/groups/swl1/swl1/fos/scripts
python3.9 seminmf-fos-boostrap.py

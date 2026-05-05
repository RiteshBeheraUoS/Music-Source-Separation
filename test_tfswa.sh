#!/bin/bash -l
#SBATCH -p ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rkb1u25@soton.ac.uk
#SBATCH --time=12:00:00
#SBATCH --output=/home/rkb1u25/logs/slurm-%j.out
#SBATCH --error=/home/rkb1u25/logs/slurm-%j.err

echo "=== Job started at $(date) ==="

# Ensure logs dir exists
mkdir -p ~/logs

# Load conda
source /iridisfs/ixsoftware/conda/miniconda-py3/etc/profile.d/conda.sh
conda activate rkb_pyEnv

echo "Checking script:"
ls ~/TFSWA_Moises.py

cd ~
python TFSWA_Moises.py evaluate \
  --data ~/moisesdb \
  --ckpt ~/runs/vocals_try2/best.pt \
  --out-dir ~/runs/TFSWA \
  --songs-json ~/runs/vocals_try2/splits.json

echo "=== Job finished at $(date) ==="
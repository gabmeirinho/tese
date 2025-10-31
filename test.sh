#!/bin/bash
#SBATCH --job-name=feature-search
#SBATCH --output=output.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --partition=compute

uv run python3 -u gggp.py
# uv run python3 -u empirical_results_new.py
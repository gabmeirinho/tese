#!/bin/bash
#SBATCH --job-name=baf_empirical_results
#SBATCH --output=output/output.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=1GB
#SBATCH --time=20:00:00

uv run python3 -u empirical_results.py

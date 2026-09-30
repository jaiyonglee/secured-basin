#!/bin/sh
 
# job name (overridden per-job by the launcher via #SBATCH --job-name)
#SBATCH -J atk_rnd
 
# output and error files (overridden per-job by the launcher)
#SBATCH -o logs_attack_random/%x.o%j
#SBATCH -e logs_attack_random/%x.e%j
 
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
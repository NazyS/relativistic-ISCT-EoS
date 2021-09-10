#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH	--mail-type=FAIL,END
#SBATCH --mail-user=nsyakovenko@gmail.com
#SBATCH --time=5-10:00:00

python3 $@

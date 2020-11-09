#!/bin/bash

labels=("vdw" "IST" "ISCT" "ISCT2")
g_array=(3 30 300 3000)
b_array=(0 1)
# mass, radius and chem. potential
m=140
R=0.39
mu=0

# submitting pions
for label in "${labels[@]}"; do

    for b in "${b_array[@]}"; do

        for g in ${g_array[@]}; do

            echo Submitting pions ${label} g: ${g} mu: ${mu} b: ${b}
            sbatch script.sh sp_of_snd.py ${m} ${label} ${g} ${mu} ${b} ${R}
            sleep 1

        done
    done
done

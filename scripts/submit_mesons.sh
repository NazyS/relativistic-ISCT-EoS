#!/bin/bash

labels=("ISCT" "ISCT2")
g_array=(30 300 3000)
masses=(30)
b_array=(0 1)
# radius and chem. potential
R=0.4
mu_array=(0 200 400 600)


# submitting light mesons
for m in "${masses[@]}"; do

    for label in "${labels[@]}"; do

        for b in "${b_array[@]}"; do

            for g in ${g_array[@]}; do

                for mu in ${mu_array[@]}; do

                    echo Submitting light mesons with m: ${m} ${label} g: ${g} mu: ${mu} b: ${b}
                    sbatch --partition $1 script.sh sp_of_snd.py ${m} ${label} ${g} ${mu} ${b} ${R}
                    sleep 1

                done
            done
        done
    done
done
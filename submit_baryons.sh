#!/bin/bash

labels=("vdw" "IST" "ISCT" "ISCT2")
g_array1=(4 4000)
g_array2=(40 400)
mu_array=(0 1000 2000)
b_array=(0 1)
# mass and radius
m=940
R=0.39

# submitting baryons
for label in "${labels[@]}"; do

    for b in "${b_array[@]}"; do

        for g in ${g_array1[@]}; do

            for mu in ${mu_array[@]}; do

                echo Submitting baryons ${label} g: ${g} mu: ${mu} b: ${b}
                sbatch script.sh sp_of_snd.py ${m} ${label} ${g} ${mu} ${b} ${R}
                sleep 1

            done
        done

        for g in ${g_array2[@]}; do

            echo Submitting baryons ${label} g: ${g} mu: ${mu_array[0]} b: ${b}
            sbatch --partition $1 script.sh sp_of_snd.py ${m} ${label} ${g} ${mu_array[0]} ${b} ${R}
            sleep 1

        done
    done
done

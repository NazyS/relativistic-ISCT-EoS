#!/bin/bash

T_array=(135 140 145 150)
# mu_array=()

# radius
# R=0.39

for T in "${T_array[@]}"; do

    # for mu in "${mu_array[@]}"; do
    for mu in `seq 0 100 1000`; do

        echo Submitting T: ${T} mu: ${mu}
        sbatch --partition $1 script.sh phase_transition.py ${T} ${mu}
        sleep 1


    done
done

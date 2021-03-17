#!/bin/bash

# T_array=(125 130 135 140 145 150)
# mu_array=()

T_array=(125 130 135)
entr_ratio_array=(11 11.31482 12 13 14 15)

# radius
# R=0.39

for T in "${T_array[@]}"; do

    # for mu in `seq 0 100 1000`; do
    for entr_ratio in "${entr_ratio_array[@]}"; do

        # echo Submitting T: ${T} mu: ${mu}
        # sbatch --partition $1 script.sh search_for_m.py ${T} ${mu}
        echo Submitting T: ${T} entr_ratio: ${entr_ratio}
        sbatch --partition $1 script.sh search_for_m_2.py ${T} ${entr_ratio}
        sleep 1

    done
done

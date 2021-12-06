import os
import sys

if __name__ == "_main__":

    partition = str(sys.argv[1])

    selected_mass = [11.48, 26.79]
    selected_b = [0.0, 1.0]
    selected_g = [(4.0, 3.0), (140.0, 1525.0)]  # fermions, bosons

    for m_bos in selected_mass:
        for b in selected_b:
            for g_F, g_B in selected_g:
                os.system(
                    "sbatch -p {} script.sh grid_table.py {} {} {} {}".format(
                        partition, g_F, g_B, m_bos, b
                    )
                )

import os
import sys

if __name__ == "__main__":

    partition = sys.argv[1]

    m_M_array = [1250.0, 1500.0, 1500.0]
    m_B_array = [2500.0, 2500.0, 3000.0]

    g_M_array = [30.0, 90.0, 300.0]
    g_B_array = [40.0, 120.0, 400.0]

    mu = 0.0

    for g_M, g_B in zip(g_M_array, g_B_array):
        for m_M, m_B in zip(m_M_array, m_B_array):
            os.system(
                "sbatch -p {} script.sh sp_of_snd_for_lattice.py {} {} {} {} {}".format(
                    partition, g_M, g_B, m_M, m_B, mu
                )
            )

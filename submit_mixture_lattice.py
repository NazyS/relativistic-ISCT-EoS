import os

m_M_array = [1250., 1500., 1500.]
m_B_array = [2500., 2500., 3000.]

g_M_array = [30., 90., 300.]
g_B_array = [40., 120., 400.]

mu = 0.

for g_M, g_B in zip(g_M_array, g_B_array):
    for m_M, m_B in zip(m_M_array, m_B_array):
        os.system(
            'sbatch -p x-men script.sh sp_of_snd_for_lattice.py {} {} {} {} {}'.format(g_M, g_B, m_M, m_B, mu)
        )
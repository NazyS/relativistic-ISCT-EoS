import os
import sys

partition = str(sys.argv[1])

selected_mass = [11.48, 26.79]
selected_b = [0., 1.]
selected_g = [
    (4., 3.),       # fermions, bosons
    (140., 1525.)
]

for m_bos in selected_mass:
    for b in selected_b:
        for g_F, g_B in selected_g:
             os.system('sbatch -p {} ../script.sh grid_table.py {} {} {} {}'.format(partition, g_F, g_B, m_bos, b))
import numpy as np
import pandas as pd
import sys
import os

sys.path.append('..')
from eos.relativistic_ISCT import Relativistic_ISCT

g_F = float(sys.argv[1])
g_B = float(sys.argv[2])
m_B = float(sys.argv[3])
b = float(sys.argv[4])

print('init for m_B: {}\t g_F: {}\t g_B: {}\t b: {}'.format(m_B, g_F, g_B, b))
print('************************************************************')

folder = '../cs_sq_fulldata/grid_tables'
filename = 'grid_table_m_B_{}_g_F_{}_g_B_{}_b_{}_.csv'.format(m_B, g_F, g_B, b)
filepath = os.path.join(folder, filename)

print('Saving to ', filepath)
print('************************************************************')

# mixture of nucleons, pions, heavy mesons and heavy baryons
eos = Relativistic_ISCT(
    components=2,
    m=[1.5*m_B, m_B],
    g=[g_F, g_B],
    eos='ISCT',
    R=0.39,
    b=b,
    )

# grid range
Trange = np.arange(100., 160., 10.)
murange = np.arange(300., 710., 10.)

data_dict = {
    'T': [],
    'mu': [],
    'p': [],
    'Sigma': [],
    'K': [],
    'entropy': [],
    'density': []
 }

for T in Trange:
    for mu in murange:
        print(' T: {}\t mu: {}'.format(T, mu))
        
        sys.stdout.flush()
        (p, Sigma, K), _ = eos.splined_root(T, mu, 0.)

        data_dict['T'].append(T)
        data_dict['mu'].append(mu)
        data_dict['p'].append(p)
        data_dict['Sigma'].append(Sigma)
        data_dict['K'].append(K)
        data_dict['entropy'].append(eos.splined_entropy(T, mu, 0.))
        data_dict['density'].append(eos.splined_density_baryon(T, mu, 0.))

        for key, val in data_dict.items():
            print(key,': ', val[-1], end='\t')
        print('\n-------------------------------------------')

df = pd.DataFrame(data_dict)
df.to_csv(filepath, index=False)
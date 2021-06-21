import numpy as np
import pandas as pd

from eos.relativistic_ISCT import Relativistic_ISCT

import os
import sys

g_M = float(sys.argv[1])
g_B = float(sys.argv[2])
m_M = float(sys.argv[3])
m_B = float(sys.argv[4])
mu = float(sys.argv[5])

print('init for mu: {}\t g_M: {}\t g_B: {}\t m_M: {}\t m_B: {}'.format(mu, g_M, g_B, m_M, m_B))
print('************************************************************')

folder = 'cs_sq_fulldata'
filepath = 'rootdata_mixture_g_M_{}_g_B_{}_m_M_{}_m_B_{}_mu_{}_.csv'.format(g_M, g_B, m_M, m_B, mu)

print('Saving to ', filepath)
print('************************************************************')

# mixture of nucleons, pions, heavy mesons and heavy baryons
eos = Relativistic_ISCT(
    components=4,
    m=[940., 140., m_M, m_B],
    g=[4., 3., g_M, g_B],
    eos='ISCT',
    R=0.39,
    b=0,    # all virial coeff. contracted
    )

Tdata = np.linspace(15., 3000., 100)
pdata = []
Sigmadata = []
Kdata = []
# entropydata = []
# densitydata = []
# cs_sq_data = []

mu = [mu, 0., 0., mu]

for T in Tdata:
    print('calculating for T: ',T)
    print('-------------------------------------------')
    sys.stdout.flush()
    (p, Sigma, K), _ = eos.splined_root(T, *mu)
    pdata.append(p)
    Sigmadata.append(Sigma)
    Kdata.append(K)
    # entropydata.append(eos.splined_entropy(T, *mu))
    # densitydata.append(eos.splined_density_baryon(T, *mu))
    # cs_sq_data.append(eos.splined_speed_of_s_sq(T, *mu))

    df = pd.DataFrame({
        'T':Tdata[:len(pdata)],
        'p':pdata,
        'Sigma':Sigmadata,
        'K':Kdata,
        # 'entropy':entropydata,
        # 'density_baryon':densitydata,
        # 'sp_of_snd_sq':cs_sq_data
    })

    df.to_csv(os.path.join(folder, filepath), index=False)
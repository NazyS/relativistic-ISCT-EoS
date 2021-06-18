import numpy as np
import pandas as pd

from eos.relativistic_ISCT import Relativistic_ISCT

import os
import sys

g_M = float(sys.argv[1])
g_B = float(sys.argv[2])
mu = float(sys.argv[3])

print('init for mu: {}\t g_M: {}\t g_B: {}'.format(mu, g_M, g_B))
print('************************************************************')

folder = 'cs_sq_fulldata'
filepath = 'cs_sq_fulldata_mixture_g_M_{}_g_B_{}_mu_{}_.csv'.format(g_M, g_B, mu)

print('Saving to ', filepath)
print('************************************************************')

# mixture of nucleons, pions, heavy mesons and heavy baryons
eos = Relativistic_ISCT(
    components=4,
    m=[940., 140., 1000., 1500.],
    g=[4., 3., g_M, g_B],
    eos='ISCT',
    R=0.39,
    b=0,    # all virial coeff. contracted
    )

Tdata = np.linspace(15., 3000., 100)
pdata = []
Sigmadata = []
Kdata = []
entropydata = []
densitydata = []
cs_sq_data = []

mu = [mu, 0., 0., mu]

for T in Tdata:
    print('calculating for T: ',T)
    print('-------------------------------------------')
    sys.stdout.flush()
    (p, Sigma, K), _ = eos.splined_root(T, *mu)
    pdata.append(p)
    Sigmadata.append(Sigma)
    Kdata.append(K)
    entropydata.append(eos.splined_entropy(T, *mu))
    densitydata.append(eos.splined_density_baryon(T, *mu))
    cs_sq_data.append(eos.splined_speed_of_s_sq(T, *mu))

    df = pd.DataFrame({
        'T':Tdata,
        'p':pdata,
        'Sigma':Sigmadata,
        'K':Kdata,
        'entropy':entropydata,
        'density_baryon':densitydata,
        'sp_of_snd_sq':cs_sq_data
    })

    df.to_csv(os.path.join(folder, filepath), index=False)
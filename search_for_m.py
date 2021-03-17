#!/usr/bin/env python

from eos.relativistic_ISCT import Relativistic_ISCT
from scipy.optimize import fsolve
import numpy as np
import pandas as pd
import sys

# inputting temperature T and baryonic chemical potential mu_b
T = float(sys.argv[1])
mu_b = float(sys.argv[2])

print('*'*50)
print('input:')
print('T: {}\t mu_b {}'.format(T, mu_b))
print('*'*50)
sys.stdout.flush()

# Bugaev, K.A., Emaus, R., Sagun, V.V. et al. Threshold Collision Energy of the QCD Phase Diagram Tricritical Endpoint. Phys. Part. Nuclei Lett. 15, 210–224 (2018). https://doi.org/10.1134/S1547477118030068
G_TOTAL = 1770.
G_FERMION = 140.                # approx 141 in article 
G_BOSON = G_TOTAL - 7./4.*G_FERMION

# Bugaev, K.A., Ivanytskyi, A.I., Oliinychenko, D.R. et al. Thermodynamically anomalous regions as a mixed phase signal. Phys. Part. Nuclei Lett. 12, 238–245 (2015). https://doi.org/10.1134/S1547477115020065
ENTR_TO_BAR_DENS_RATIO = 11.31482


def search_for_low_m(m, T, mu_b):
    eos = Relativistic_ISCT(m=m, R=0.39, b=0.,  components=2, eos='ISCT', g=[G_FERMION, G_BOSON])
    entr_per_bar_dens = eos.entropy(T, mu_b, 0.)/eos.density_baryon(T, mu_b, 0.)
    return entr_per_bar_dens - ENTR_TO_BAR_DENS_RATIO

m, info, solved, msg = fsolve(search_for_low_m, 30., args=(T, mu_b), full_output=1)

if solved!=1:
    print(
        'T: {}\t mu_b: {}\t m: {}'.format(T, mu_b, msg)
    )
    m = np.nan 
    entr_ratio = np.nan
    dev = np.nan
else:
    print(
        'T: {}\t mu_b: {}\t m: {}'.format(T, mu_b, m)
    )
    print(msg)
    print('info', info)

    dev = info['fvec']
    entr_ratio = dev + ENTR_TO_BAR_DENS_RATIO


filename = 'low_m_search_T_{}_mu_b_{}_.csv'.format(T, mu_b)

df = pd.DataFrame({
    'T':T,
    'mu':mu_b,
    'm':m,
    'entr_per_bar_dens':entr_ratio,
    'dev':dev,
})

df.to_csv(filename, index=False)
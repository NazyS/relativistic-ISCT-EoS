from eos.relativistic_ISCT import Relativistic_ISCT
import numpy as np
import pandas as pd
import sys
import os

# inputting temperature T and entropy ratio
T = float(sys.argv[1])
m_bos = float(sys.argv[2])
g_fer = float(sys.argv[3])
g_bos = float(sys.argv[4])

print('*'*50)
print('input:')
print('T: {}\t m_bos: {}\t g_fer: {}\t g_bos: {}'.format(T, m_bos, g_fer, g_bos))
print('*'*50)
sys.stdout.flush()

eos = Relativistic_ISCT(m=[1.5*m_bos, m_bos], R=0.39, b=0.,  components=2, eos='ISCT', g=[g_fer, g_bos])

mu_data = np.linspace(10., 1000., 100)

p_data = []
Sigma_data = []
K_data = []

k1_data = []
k2_data = []
k3_data = []
k_lin_ratio = []
k_sq_ratio = []

for mu in mu_data:
    print('mu: {}'.format(mu))
    sys.stdout.flush()

    p, Sigma, K = eos.p_eq(T, mu, 0., format='full')
    p_data.append(p)
    Sigma_data.append(Sigma)
    K_data.append(K)

    k1_data.append(eos.cumulant_per_vol(1, 0, T, mu, 0.))
    k2_data.append(eos.cumulant_per_vol(2, 0, T, mu, 0.))
    k3_data.append(eos.cumulant_per_vol(3, 0, T, mu, 0.))
    k_lin_ratio.append(eos.cumul_lin_ratio(T, mu, 0.))
    k_sq_ratio.append(eos.cumul_sq_ratio(T, mu, 0.))


df = pd.DataFrame({
    'mu':mu_data,
    'p':p_data,
    'Sigma':Sigma_data,
    'K':K_data,
    'k1_per_V':k1_data,
    'k2_per_V':k2_data,
    'k3_per_V':k3_data,
    'k_lin_ratio':k_lin_ratio,
    'k_sq_ratio':k_sq_ratio
})

print(df)

filename = 'cumul_data_T_{}_m_bos_{}_g_fer_{}_g_bos_{}_.csv'.format(T, m_bos, g_fer, g_bos)
folder = 'cs_sq_fulldata/cumuls'
if not os.path.exists(folder):
    os.makedirs(folder)

df.to_csv(
    os.path.join(folder, filename),
    index=False
)
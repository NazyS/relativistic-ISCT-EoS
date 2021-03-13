#!/usr/bin/env python
# coding: utf-8

# In[1]:


from eos.relativistic_ISCT import Relativistic_ISCT
from scipy.optimize import fsolve
import numpy as np

# Bugaev, K.A., Emaus, R., Sagun, V.V. et al. Threshold Collision Energy of the QCD Phase Diagram Tricritical Endpoint. Phys. Part. Nuclei Lett. 15, 210–224 (2018). https://doi.org/10.1134/S1547477118030068
G_TOTAL = 1770.
G_FERMION = 140.                # approx 141 in article 
G_BOSON = G_TOTAL - 7./4.*G_FERMION

# Bugaev, K.A., Ivanytskyi, A.I., Oliinychenko, D.R. et al. Thermodynamically anomalous regions as a mixed phase signal. Phys. Part. Nuclei Lett. 12, 238–245 (2015). https://doi.org/10.1134/S1547477115020065
ENTR_TO_BAR_DENS_RATIO = 11.31482


# In[ ]:





# In[2]:


def search_for_low_m(m, T, mu_b):
    eos = Relativistic_ISCT(m=m, R=0.39,  components=2, eos='ISCT', g=[G_FERMION, G_BOSON])
    entr_per_bar_dens = eos.entropy(T, mu_b, 0.)/eos.density_baryon(T, mu_b, 0.)
    return entr_per_bar_dens - ENTR_TO_BAR_DENS_RATIO


# In[36]:


for T in np.arange(135., 155., 5.):
    for mu_b in np.linspace(0., 1000., 100.):

        try:
            m = fsolve(search_for_low_m, 30., args=(T, mu_b))
            print(
                'T: {}\t mu_b: {}\t m: {}'.format(T, mu_b, m)
            )
        except: 
            print(
                'T: {}\t mu_b: {}\t mass not found'.format(T, mu_b)
            )


# In[ ]:


m


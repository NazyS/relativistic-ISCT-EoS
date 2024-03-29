import os
import sys
import numpy as np
import pandas as pd

import context
from main import get_changable_dx
from eos.relativistic_ISCT import Relativistic_ISCT

if __name__ == "__main__":

    # parameters init
    # we are working with:
    # baryons : m = 940, R = 0.39
    # pions :   m = 140, R = 0.39
    # light mesons: m = 20, 25, 30, R = 0.4
    m = float(sys.argv[1])
    label = str(sys.argv[2])
    g = float(sys.argv[3])
    mu = float(sys.argv[4])
    b = float(sys.argv[5])
    R = float(sys.argv[6])

    print(
        "init for {} with g: {}, mu: {}, b: {}, m: {}, R: {}".format(
            label, g, mu, b, m, R
        )
    )
    print("************************************************************")

    if m == 940:
        particle_type = "baryons"
    elif m == 140.0:
        particle_type = "pions"
    elif m == 20.0 or m == 25.0 or m == 30.0:
        particle_type = "ligth_mes_m_{}".format(m)
    else:
        particle_type = "m_{}".format(m)

    folder = "../cs_sq_fulldata"
    filepath = "cs_sq_fulldata_{}_b_{}_{}_g_{}_R_{}_mu_{}_.csv".format(
        particle_type, b, label, g, R, mu
    )

    eos = Relativistic_ISCT(m=m, g=g, eos=label, R=R, b=b)

    Tdata = np.linspace(2.0, 3000.0, 200)
    pdata = []
    Sigmadata = []
    Kdata = []
    entropydata = []
    densitydata = []
    cs_sq_data = []

    mu = [mu]

    for T in Tdata:
        print("calculating for T: ", T)
        print("-------------------------------------------")
        sys.stdout.flush()
        p, Sigma, K = eos.p_eq(T, *mu, format="all")
        pdata.append(p)
        Sigmadata.append(Sigma)
        Kdata.append(K)
        entropydata.append(eos.entropy(T, *mu, dx=get_changable_dx(T)))
        densitydata.append(eos.density_baryon(T, *mu, dx=get_changable_dx(T)))
        cs_sq_data.append(eos.speed_of_s_sq(T, *mu, dx=get_changable_dx(T)))

    df = pd.DataFrame(
        {
            "T": Tdata,
            "p": pdata,
            "Sigma": Sigmadata,
            "K": Kdata,
            "entropy": entropydata,
            "density_baryon": densitydata,
            "sp_of_snd_sq": cs_sq_data,
        }
    )

    df.to_csv(os.path.join(folder, filepath), index=False)

    print("Saved to ", filepath)
    print("************************************************************")

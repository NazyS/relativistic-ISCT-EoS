import sys
import os
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

import context
from eos.relativistic_ISCT import Relativistic_ISCT

if __name__ == "__main__":

    m_bos = float(sys.argv[1])
    g_fer = float(sys.argv[2])
    g_bos = float(sys.argv[3])
    row = int(sys.argv[4])

    cumuls_exp = pd.read_csv("../cs_sq_fulldata/cumuls/cumuls_star+hades_digit_.csv")
    mu = cumuls_exp.iloc[row]["mu"]
    # cumul = cumuls_exp.iloc[row]['cumul_sq']
    cumul = cumuls_exp.iloc[row]["cumul_lin"]

    print("*" * 50)
    print("input:")
    print("mu: {} \t m_bos: {}\t g_fer: {}\t g_bos: {}".format(mu, m_bos, g_fer, g_bos))
    print("*" * 50)
    sys.stdout.flush()

    eos = Relativistic_ISCT(
        m=[1.5 * m_bos, m_bos],
        R=0.39,
        b=0.0,
        components=2,
        eos="ISCT",
        g=[g_fer, g_bos],
    )

    def match_temp(T, mu, cumul):
        # T = float(T)
        # return (eos.splined_cumul_sq_ratio(T, mu, 0.) - cumul_sq)**2
        return (eos.splined_cumul_lin_ratio(T, mu, 0.0) - cumul) ** 2

    # res = minimize(match_temp, 175., args=(mu, cumul_sq_ratio))
    res = minimize_scalar(
        match_temp, args=(mu, cumul), method="bounded", bounds=(20.0, 250.0)
    )
    print(res)
    T = res.x

    df = pd.DataFrame(
        {
            "T": [T],
            "mu": [mu],
            "k1_data": [eos.splined_cumul_per_vol(1, 0, T, mu, 0.0)],
            "k2_data": [eos.splined_cumul_per_vol(2, 0, T, mu, 0.0)],
            "k3_data": [eos.splined_cumul_per_vol(3, 0, T, mu, 0.0)],
            "cumul_lin_ratio": [eos.splined_cumul_lin_ratio(T, mu, 0.0)],
            "cumul_sq_ratio": [eos.splined_cumul_sq_ratio(T, mu, 0.0)],
        }
    )

    filename = "fo_temp_search_mu_{}_m_bos_{}_g_fer_{}_g_bos_{}_.csv".format(
        mu, m_bos, g_fer, g_bos
    )
    folder = "../cs_sq_fulldata/cumuls"

    if not os.path.exists(folder):
        os.makedirs(folder)

    df.to_csv(os.path.join(folder, filename), index=False)

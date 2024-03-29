import sys
import numpy as np
import pandas as pd
from scipy.optimize import fsolve

import context
from eos.relativistic_ISCT import Relativistic_ISCT


if __name__ == "__main__":

    # inputting temperature T and entropy ratio
    T = float(sys.argv[1])
    entr_ratio = float(sys.argv[2])
    BAR_DENS = float(sys.argv[3])

    print("*" * 50)
    print("input:")
    print("T: {}\t entr_ratio: {}\t bar_dens: {}".format(T, entr_ratio, BAR_DENS))
    print("*" * 50)
    sys.stdout.flush()

    # Bugaev, K.A., Emaus, R., Sagun, V.V. et al. Threshold Collision Energy of the QCD Phase Diagram Tricritical Endpoint. Phys. Part. Nuclei Lett. 15, 210–224 (2018). https://doi.org/10.1134/S1547477118030068
    G_TOTAL = 1770.0
    # G_FERMION = 140.                # approx 141 in article
    G_FERMION = 126.0  # 10% lower
    G_BOSON = G_TOTAL - 7.0 / 4.0 * G_FERMION

    # Bugaev, K.A., Ivanytskyi, A.I., Oliinychenko, D.R. et al. Thermodynamically anomalous regions as a mixed phase signal. Phys. Part. Nuclei Lett. 12, 238–245 (2015). https://doi.org/10.1134/S1547477115020065
    # ENTR_TO_BAR_DENS_RATIO = 11.31482

    # fixed baryon density for mixed hadron phase
    # BAR_DENS = 13.

    def search_for_low_m(root, T, entr_ratio):
        m, mu_b = root
        eos = Relativistic_ISCT(
            m=[1.5 * m, m],
            R=0.39,
            b=0.0,
            components=2,
            eos="ISCT",
            g=[G_FERMION, G_BOSON],
        )
        bar_dens = eos.density_baryon(T, mu_b, 0.0)
        entr_per_bar_dens = eos.entropy(T, mu_b, 0.0) / bar_dens
        return entr_per_bar_dens - entr_ratio, bar_dens - BAR_DENS

    root, info, solved, msg = fsolve(
        search_for_low_m, [10.0, 300.0], args=(T, entr_ratio), full_output=1
    )
    m, mu_b = root

    print(msg)
    print("info", info)

    if solved != 1:
        m = np.nan
        mu_b = np.nan

    else:
        print(
            "T: {}\t entr ratio: {}\t m: {}\t mu_b: {}".format(T, entr_ratio, m, mu_b)
        )

    filename = "low_m_search_T_{}_entr_ratio_{}_bar_dens_{}_nm_lgf.csv".format(
        T, entr_ratio, BAR_DENS
    )

    df = pd.DataFrame(
        {
            "T": [T],
            "mu": [mu_b],
            "m": [m],
            "entr_per_bar_dens": [entr_ratio],
            "bar_dens": [BAR_DENS],
        }
    )

    df.to_csv(filename, index=False)

import sys
import os
import pandas as pd

import context
from eos.relativistic_ISCT import Relativistic_ISCT

if __name__ == "__main__":

    m_bos = float(sys.argv[1])
    g_fer = float(sys.argv[2])
    g_bos = float(sys.argv[3])

    print("*" * 50)
    print("input:")
    print("m_bos: {}\t g_fer: {}\t g_bos: {}".format(m_bos, g_fer, g_bos))
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

    for dataname in ["exp_fo", "matched"]:
        print(dataname)

        phase_diagr_df = pd.read_csv(
            os.path.join(
                "../cs_sq_fulldata/cumuls", "T_mu_phase_diagr_{}_.csv".format(dataname)
            )
        )

        mu_data = phase_diagr_df["mu"].to_numpy()
        T_data = phase_diagr_df["T"].to_numpy()

        p_data = []
        Sigma_data = []
        K_data = []

        k1_data = []
        k2_data = []
        k3_data = []
        k_lin_ratio = []
        k_sq_ratio = []

        for mu, T in zip(mu_data, T_data):
            print("T: {}\t mu: {}".format(T, mu))
            sys.stdout.flush()

            (p, Sigma, K), _ = eos.splined_root(T, mu, 0.0)
            p_data.append(p)
            Sigma_data.append(Sigma)
            K_data.append(K)

            k1_data.append(eos.splined_cumul_per_vol(1, 0, T, mu, 0.0))
            k2_data.append(eos.splined_cumul_per_vol(2, 0, T, mu, 0.0))
            k3_data.append(eos.splined_cumul_per_vol(3, 0, T, mu, 0.0))
            k_lin_ratio.append(eos.splined_cumul_lin_ratio(T, mu, 0.0))
            k_sq_ratio.append(eos.splined_cumul_sq_ratio(T, mu, 0.0))

        df = pd.DataFrame(
            {
                "T": T_data,
                "mu": mu_data,
                "p": p_data,
                "Sigma": Sigma_data,
                "K": K_data,
                "k1_per_V": k1_data,
                "k2_per_V": k2_data,
                "k3_per_V": k3_data,
                "k_lin_ratio": k_lin_ratio,
                "k_sq_ratio": k_sq_ratio,
            }
        )

        filename = "splined_cumul_data_{}_m_bos_{}_g_fer_{}_g_bos_{}_.csv".format(
            dataname, m_bos, g_fer, g_bos
        )
        folder = "../cs_sq_fulldata/cumuls"
        if not os.path.exists(folder):
            os.makedirs(folder)

        df.to_csv(os.path.join(folder, filename), index=False)

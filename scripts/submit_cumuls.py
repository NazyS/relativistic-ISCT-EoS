import sys
import os
import pandas as pd

if __name__ == "__main__":

    partition = str(sys.argv[1])

    selected_df = pd.read_csv("../cs_sq_fulldata/cumuls/low_m_selected_points_.csv")

    g_vals = [[140.0, 1525.0], [140.0, 1700.0]]

    for i in range(len(selected_df)):
        row = selected_df.iloc[i]
        # T = row['T']
        m_bos = row["m"]

        for g_pairs in g_vals:
            g_fer, g_bos = g_pairs

            # os.system('sbatch -p {} script.sh cumulants.py {} {} {} {}'.format(partition, T, m_bos, g_fer, g_bos))
            os.system(
                "sbatch -p {} script.sh cumulants.py {} {} {}".format(
                    partition, m_bos, g_fer, g_bos
                )
            )

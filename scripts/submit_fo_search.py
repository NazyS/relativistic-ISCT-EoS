import sys
import os
import pandas as pd

if __name__ == "__main__":

    partition = sys.argv[1]

    df = pd.read_csv("../cs_sq_fulldata/cumuls/cumuls_star+hades_digit_.csv")

    m_bos = 29.93382977815157
    g_fer = 140.0
    g_bos = 1525.0

    for row in range(len(df)):
        os.system(
            "sbatch --partition {} script.sh freeze-out_search.py {} {} {} {}".format(
                partition, m_bos, g_fer, g_bos, row
            )
        )

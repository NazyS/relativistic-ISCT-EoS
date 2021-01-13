import numpy as np

def clean_outliers_and_mask(df, threshold=5e-4, repeats=4, window=3):

    for _ in range(repeats):
        cs_data = df['sp_of_snd_sq']
        diff = np.abs((cs_data - cs_data.fillna(method='bfill').fillna(method='ffill')
.rolling(window, center=True).median()))
        # diff = np.abs((cs_data - cs_data.rolling(window, center=True).mean()))
        outliers = diff > threshold
        
        mask = np.invert(outliers)
        df = df[mask]

    nan_mask = np.invert(df['sp_of_snd_sq'].isna())
    df = df[nan_mask]

    return df

def evenly_spaced_data(df, interval=300, points=20):
    Tdata = df['T'].to_numpy()
    cs_data = np.sqrt( df['sp_of_snd_sq'].to_numpy() )

    interval = interval if interval else (Tdata.max() - Tdata.min())/(points + 1)
    Tplot = [0.]
    cs_plot = [0.]

    for i in range(len(Tdata)):
        if Tdata[i] - Tplot[-1] > interval:
            Tplot.append(Tdata[i])
            cs_plot.append(cs_data[i])
    
    return Tplot[1:], cs_plot[1:]

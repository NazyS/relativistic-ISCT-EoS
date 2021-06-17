import numpy as np


def clean_outliers_and_mask(df, label='sp_of_snd_sq', threshold=5e-4, repeats=4, window=3):

    for _ in range(repeats):
        data = df[label]
        diff = np.abs((data - data.fillna(method='bfill').fillna(method='ffill').rolling(window, center=True).median()))
        outliers = diff > threshold

        mask = np.invert(outliers)
        df = df[mask]

    nan_mask = np.invert(df[label].isna())
    df = df[nan_mask]

    return df


def evenly_spaced_data(xdata, ydata, interval=300, points=10, xscale='log'):
    xdata = np.array(xdata)
    ydata = np.array(ydata)

    if xscale=='log':
        interval = interval if interval else (xdata.max() - xdata.min())/(points + 1)
        interval = interval/points
        delta = (xdata.max()/interval)**(1./(points+2))
    else:    
        interval = interval if interval else (xdata.max() - xdata.min())/(points + 1)
        delta = 1.

    xplot = [0.]
    yplot = [0.]

    for i in range(len(xdata)):

        if xdata[i] - xplot[-1] > interval:

            xplot.append(xdata[i])
            yplot.append(ydata[i])
            interval = interval*delta

    return xplot[1:], yplot[1:]


def get_parameters_from_particle_type(particle_type):
    # we are working with:
    # baryons : m = 940, R = 0.39
    # pions :   m = 140, R = 0.39
    # light mesons: m = 20, 25, 30, R = 0.4
    if particle_type == 'baryons':
        m = 940.
        R = 0.39
    elif particle_type == 'pions':
        m = 140.
        R = 0.39
    elif 'ligth_mes' in particle_type:
        R = 0.4
        m = float(particle_type[len('ligth_mes_m_'):])    # cutting string to get m value
    else:
        raise Exception('Wrong particle type')

    return m, R

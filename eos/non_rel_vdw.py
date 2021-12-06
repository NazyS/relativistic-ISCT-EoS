import numpy as np

from main import Eq_of_state, partial_derivative


class Non_Rel_VdW(Eq_of_state):
    """
    Non-relativistic gas with VdW pressure.
    Was used for comparison with results for relativistic ISCT in the low temperature T regime.
    """

    def __init__(self, a=1.0, g=4.0, m=940.0, R=0.39):
        self.a = a
        self.g = g
        self.m = m
        self.R = R

        # excluded volume
        self.b = 4.0 * 4.0 / 3.0 * np.pi * self.R ** 3

        # description of eos
        self.num_of_components = 1
        self.num_of_eq_in_eos = 1

    # no antibaryons included
    def EoS(self, root, T, mu):
        p = root
        # return self.a*self.g*T**(5./2.)*np.exp((mu - self.m)/T - self.b*p/T) - p
        return (
            2
            * self.a
            * self.g
            * T ** (5.0 / 2.0)
            * np.exp(-self.m / T - self.b * p / T)
            * np.cosh(mu / T)
            - p
        )

    def entr_analyt(self, T, mu):
        p = self.p_eq(T, mu)
        num = p / T * (5.0 / 2.0 + self.b * p / T - (mu - self.m) / T)
        denum = 1 + self.b * p / T
        return num / denum

    def dens_analyt(self, T, mu):
        p = self.p_eq(T, mu)
        return p / T / (1 + self.b * p / T)

    def energy_analyt(self, T, mu):
        p = self.p_eq(T, mu)
        return T * self.entropy(T, mu) + mu * self.dens_analyt(T, mu) - p

    def analyt_sp_of_snd_sq(self, T, mu, **kwargs):
        p = self.p_eq(T, mu, **kwargs)

        num = (1 + self.b * p / T) ** 2
        denum = 3.0 / 2.0 + 3.0 / 5.0 * (self.m / T + self.b * p / T)
        return num / denum

    def analyt_sp_of_snd_sq_new(self, T, mu, **kwargs):
        p = self.p_eq(T, mu, **kwargs)

        sigma = 5.0 / 2.0 + self.b * p / T - (mu - self.m) / T
        rho = p / T / (1 + self.b * p / T)
        # energy = p*(3./2. + self.m/T)/(1 + self.b*p/T)
        energy = T * sigma * rho + mu * rho - p
        d_mu_dt = -(self.b * sigma * rho - self.b * p / T + (mu - self.m) / T) / (
            self.b * rho - 1.0
        )

        num = sigma * rho + rho * d_mu_dt
        denum = (
            sigma * rho * energy / p
            - p * self.m / T ** 2 / (1.0 + self.b * p / T)
            - self.b * energy * (sigma * rho - p / T) / T / (1 + self.b * p / T)
            + (rho * energy / p - self.b * rho * energy / T / (1 + self.b * p / T))
            * d_mu_dt
        )

        return num / denum


class Non_Rel_VdW_with_antibaryons(Eq_of_state):
    """
    Non-relativistic gas with VdW pressure
    used to check low T results for relativistic ISCT
    """

    def __init__(self, a=1.0, g=4.0, m=940.0, R=0.39):
        self.a = a
        self.g = g
        self.m = m
        self.R = R

        # excluded volume
        self.b = 4.0 * 4.0 / 3.0 * np.pi * self.R ** 3

        # description of eos
        self.num_of_components = 1
        self.num_of_eq_in_eos = 1

    # antibaryons included
    def EoS(self, root, T, mu):
        p = root
        return (
            2.0
            * self.a
            * self.g
            * T ** (5.0 / 2.0)
            * np.exp(-self.m / T - self.b * p / T)
            * np.cosh(mu / T)
            - p
        )

    def entr_analyt(self, T, mu):
        p = self.p_eq(T, mu)
        num = (
            p / T * (5.0 / 2.0 - mu / T * np.tanh(mu / T) + self.m / T + self.b * p / T)
        )
        denum = 1 + self.b * p / T
        return num / denum

    def dens_analyt(self, T, mu):
        p = self.p_eq(T, mu)
        return p * np.tanh(mu / T) / T / (1 + self.b * p / T)

    def energy_analyt(self, T, mu):
        p = self.p_eq(T, mu)
        return p * (3.0 / 2.0 + self.m / T) / (1.0 + self.b * p / T)

    def sigma_analyt(self, T, mu):
        p = self.p_eq(T, mu)
        num = 5.0 / 2.0 - mu / T * np.tanh(mu / T) + self.m / T + self.b * p / T
        denum = np.tanh(mu / T)
        res = num / denum
        if np.isposinf(res) or np.isneginf(res):
            return 0.0
        else:
            return res

    def d_energy_dT_analyt(self, T, mu):
        p = self.p_eq(T, mu)
        return (
            self.energy_analyt(T, mu) * self.entr_analyt(T, mu) / p
            - p * self.m / T ** 2 / (1 + self.b * p / T)
            - self.energy_analyt(T, mu)
            * self.b
            * (self.entr_analyt(T, mu) - p / T)
            / T
            / (1 + self.b * p / T)
        )

    def d_energy_dmu_analyt(self, T, mu):
        p = self.p_eq(T, mu)
        return self.energy_analyt(T, mu) * self.dens_analyt(
            T, mu
        ) / p - self.energy_analyt(T, mu) * self.b * self.dens_analyt(T, mu) / T / (
            1 + self.b * p / T
        )

    def dmu_dt_analyt(self, T, mu):
        p = self.p_eq(T, mu)

        num = (
            mu / T ** 2 * np.tanh(mu / T)
            + mu ** 2 / T ** 3 / np.cosh(mu / T) ** 2
            - self.m / T ** 2
            + self.b * self.entr_analyt(T, mu) / T
            - self.b * p / T ** 2
        ) / np.tanh(mu / T) + mu / T ** 2 * self.sigma_analyt(T, mu) / np.tanh(
            mu / T
        ) / np.cosh(
            mu / T
        ) ** 2
        denum = (
            -np.tanh(mu / T)
            - mu / T / np.cosh(mu / T) ** 2
            + self.b * self.dens_analyt(T, mu)
        ) / (T * np.tanh(mu / T)) - self.sigma_analyt(T, mu) / T / np.tanh(
            mu / T
        ) / np.cosh(
            mu / T
        ) ** 2
        res = -num / denum
        if np.isnan(res):
            return 0.0
        else:
            return res

    def analyt_sp_of_snd_sq(self, T, mu):
        dmu_dt = self.dmu_dt_analyt(T, mu)

        num = self.entr_analyt(T, mu) + self.dens_analyt(T, mu) * dmu_dt
        denum = (
            self.d_energy_dT_analyt(T, mu) + self.d_energy_dmu_analyt(T, mu) * dmu_dt
        )
        return num / denum

    def analyt_sp_of_snd_sq2(self, T, mu, **kwargs):
        def dmu_dt_analyt(T, mu, **kwargs):
            num = partial_derivative(self.sigma_analyt, 0, [T, mu], **kwargs)
            denum = partial_derivative(self.sigma_analyt, 1, [T, mu], **kwargs)
            res = -num / denum
            if np.isnan(res):
                return 0.0
            else:
                return res

        dmu_dt = dmu_dt_analyt(T, mu, **kwargs)

        num = self.entr_analyt(T, mu) + self.dens_analyt(T, mu) * dmu_dt
        denum = (
            partial_derivative(self.energy_analyt, 0, [T, mu], **kwargs)
            + partial_derivative(self.energy_analyt, 1, [T, mu], **kwargs) * dmu_dt
        )
        return num / denum

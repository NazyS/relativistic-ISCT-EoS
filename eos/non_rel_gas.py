import numpy as np
from scipy.misc import derivative

from main import Eq_of_state, hbar


class Non_Rel_Gas(Eq_of_state):
    """
    Equation of state of the non relativistic gas
    """

    def __init__(self, g=3.0, m=140.0):
        super().__init__(num_of_eq_in_eos=1, num_of_components=1)
        self.g = g
        self.m = m

    # def p_term(self, T, mu):
    #     return self.g * T * (self.m*T/(2*np.pi*hbar**2))**1.5 * np.exp((mu-self.m)/T)

    # def p_total(self, T, mu):
    #     return self.p_term(T, mu) + self.p_term(T, -mu)

    def p_total(self, T, mu):
        return (
            self.g
            * T
            * (self.m * T / (2 * np.pi * hbar ** 2)) ** 1.5
            * np.exp((-self.m) / T)
            * np.cosh(mu / T)
        )

    def EoS(self, p, T, mu):
        return p - self.p_total(T, mu)

    def entr_analyt(self, T, mu):
        p = self.p_eq(T, mu)
        return p / T * (5.0 / 2.0 - mu / T * np.tanh(mu / T) + self.m / T)

    def dens_analyt(self, T, mu):
        p = self.p_eq(T, mu)
        return p / T * np.tanh(mu / T)

    def energy_analyt(self, T, mu):
        p = self.p_eq(T, mu)
        return p * (1.5 + self.m / T)

    # analytical expression for mu = 0 and no antibaryons
    def Xi_with_log(self, T):
        return derivative(lambda T: np.log(self.entropy(T, 0)), T, dx=1e-6)

    def Xi(self, T, **kwargs):
        return derivative(
            lambda T: self.entropy(T, 0, **kwargs), T, **kwargs
        ) / self.entropy(T, 0, **kwargs)

    def speed_of_s_sq_Xi(self, T, **kwargs):
        return 1 / T / self.Xi(T, **kwargs)

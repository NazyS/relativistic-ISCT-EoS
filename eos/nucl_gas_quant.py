import numpy as np
from scipy.integrate import quad

from main import Eq_of_state, hbar


class Nucl_gas_quant(Eq_of_state):
    """
    Equation of state of quantum nucleon gas
    """

    def __init__(
        self,
        b=1.0,
        m=938.0,
        g=4.0,
    ):
        super().__init__(num_of_eq_in_eos=1, num_of_components=1)

        self.b = b
        self.m = m
        self.g = g

    def EoS(self, p, T, mu):
        def E(k):
            return np.sqrt(k ** 2 + self.m ** 2)

        def integr_func(k):
            return k ** 2 / (np.exp((-mu + E(k) + self.m * self.b * p / E(k)) / T) + 1)

        return (
            self.g
            * T
            * (4 * np.pi)
            / (2 * np.pi * hbar) ** 3
            * quad(
                integr_func,
                0,
                np.inf,
            )[0]
            - p
        )

import numpy as np
from scipy.integrate import quad

from main import Eq_of_state, hbar


# # Nucleon gas: classical

class Nucl_gas_classical(Eq_of_state):

    b = 1.
    m = 938.
    g = 4.

    num_of_eq_in_eos = 1

    def EoS(self, p, T, mu):
        def E(k):
            return np.sqrt(k**2 + self.m**2)

        def integr_func(k): return k**2 * \
            np.exp((mu - E(k) - self.m*self.b*p/E(k))/T)

        return self.g*T*(4*np.pi)/(2*np.pi*hbar)**3 * quad(integr_func, 0, np.inf, )[0] - p
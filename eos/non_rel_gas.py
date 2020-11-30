import numpy as np
from scipy.misc import derivative

from main import Eq_of_state, hbar


# # Non relativistic gas

class Non_Rel_Gas(Eq_of_state):

    def __init__(self, g=3., m=140.):
        self.g = g
        self.m = m

    num_of_eq_in_eos = 1

    def EoS(self, p, T, mu):
        return self.g * T * (self.m*T/(2*np.pi*hbar**2))**1.5 * np.exp(-self.m/T) - p

    def Xi_with_log(self, T, mu):
        return derivative(lambda T: np.log(self.entropy(T, mu)), T, dx=1e-6)

    def Xi(self, T, mu):
        return derivative(lambda T: self.entropy(T, mu), T, dx=1e-6)/self.entropy(T, mu)

    def speed_of_s_sq_Xi(self, T, mu):
        return 1/T/self.Xi(T, mu)
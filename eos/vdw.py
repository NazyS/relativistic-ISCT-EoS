import numpy as np
from main import Eq_of_state, phi

# # Van der Waals Equation Subclass
#               can be exactly reproduced with modified_exc_volume subclass
#               with parameters:    a1 = 1.,    q1 = 0.5,   a2 = 0.,    q2 and p0 - arbitrary

class VdW(Eq_of_state):

    def __init__(self, R=0.4, p0=20., m=940.):
        self.m = m

        self.R = R  # hard-core radii
        self.p0 = p0
        self.b = self.R**3 * 4*np.pi/3 * 4    # excluded volume per particle

    num_of_components = 1
    num_of_eq_in_eos = 1

    def EoS(self, p, T, mu):
        return T*phi(T, self.m) * np.exp(mu/T - self.b*p/T/(1. + np.sqrt(p/self.p0))) - p

    def parameters_string(self):
        return 'm = {:.2f} MeV, R = {:.2f} fm, $p_0$ = {:.2f} $MeV/fm^3$'.format(self.m, self.R, self.p0)
import numpy as np
from main import Eq_of_state, phi

# # Eq of state with $f(p)$ excluded volume function Subclass

class f_p_exc_vol(Eq_of_state):

    def __init__(self, q1=0.5, a1=0.06, q2=0.25, a2=0.01, p0=20., R=0.4, m=940.):
        self.m = m

        # super().__init__(self.EoS)
        self.q1 = q1
        self.a1 = a1
        self.q2 = q2  # f(p) function parameters
        self.a2 = a2
        self.p0 = p0

        self.R = R  # hard-core radii

    num_of_eq_in_eos = 1

    def f(self, p):
        return 1 / (1 + self.a1*(p/self.p0)**self.q1 + self.a2*(p/self.p0)**self.q2)

    # def f_deriv_symb(self, p):
    #     return - self.f(p)**2 * ( self.a1*self.q1/self.p0*(p/self.p0)**(self.q1-1) + self.a2*self.q2/self.p0*(p/self.p0)**(self.q2-1) )

    # def f_deriv_num(self,p):
    #     return derivative(self.f, p, dx=1e-6)

    # R = 0.4

    def b(self):
        return 4 * self.R**3 * 4*np.pi/3

    def EoS(self, p, T, mu):
        return T * phi(T, self.m) * np.exp((mu - self.b()*self.f(p)*p)/T) - p

    def parameters_string(self):
        return 'R = {:.2f} fm, $p_0$ = {:.2f} $MeV/fm^3$\n$q_1$ = {:.2f}, $a_1$ = {:.2f}, $q_2$ = {:.2f}, $a_2$ = {:.2f}'.format(self.R, self.p0, self.q1, self.a1, self.q2, self.a2)

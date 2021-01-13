import numpy as np
import scipy.integrate as integrate
from scipy.optimize import fsolve

from main import Eq_of_state, hbar


class Relativistic_ISCT(Eq_of_state):
    def __init__(self, alpha=1., beta=1., m=940., R=0.5, g=4., alpha_p=1., beta_p=1., a=0.775, b=1., eos=None, components=1):
        self.num_of_components = components
        self.m = self.make_parameter_array(m)
        self.R = self.make_parameter_array(R)
        self.g = self.make_parameter_array(g)
        self.b = b

        if eos == 'vdw':
            self.alpha = 1.
            self.beta = 1.
            self.alpha_p = 1.
            self.beta_p = 1.
        elif eos == 'IST':
            self.alpha = 1.245
            self.beta = 1.
            self.alpha_p = 2.
            self.beta_p = 0.
        elif eos == 'ISCT':
            self.alpha = 1.07
            self.beta = 3.76
            self.alpha_p = 1.14
            self.beta_p = 1.52
        elif eos == 'ISCT2':
            self.alpha = 1.14
            self.beta = 3.37
            self.alpha_p = 0.69*2.
            self.beta_p = (1.-0.69)*2.
        else:
            self.alpha = alpha
            self.beta = beta
            self.alpha_p = alpha_p
            self.beta_p = beta_p

        self.a = a

        self.v = 4./3.*np.pi*self.R**3
        self.s = 4.*np.pi*self.R**2
        self.c = 4.*np.pi*self.R

        self.num_of_eq_in_eos = 3

    # function to make parameter array of expected size
    def make_parameter_array(self, in_values):
        input = np.array([in_values]).flatten()
        if len(input) == self.num_of_components:
            return input
        else:
            if len(input) == 1:
                return np.repeat(input, self.num_of_components)
            else:
                raise Exception('Wrong parameters format')

    @staticmethod
    def delta_c1(theta):
        return np.cos(theta) - 1

    @staticmethod
    def delta_c2(theta):
        return np.cos(theta)**2 - 1

    def gamma(self, comp, k):
        return np.sqrt(self.m[comp]**2 + k**2)/self.m[comp]

    # volume operator
    def v_operator(self, comp, k, theta):
        return self.v[comp]/self.gamma(comp, k)

    # surface operator
    def s_operator(self, comp, k, theta):
        return self.s[comp]*(1./2./self.gamma(comp, k)*(1. + self.delta_c1(theta)/2.) + self.a/2.*np.abs(np.sin(theta)))

    # curvature operator
    def c_operator(self, comp, k, theta):
        return self.c[comp]/2./self.gamma(comp, k)*(1. + self.delta_c1(theta)/2. + self.delta_c2(theta)/6.)

    # integration over momentum term in thermal density
    def momentum_part(self, comp, x, T):  # after change of variables k -> m tan( pi/4 (x + 1)) and integration limits (0, inf) -> (-1, 1)
        k = self.m[comp] * np.tan(np.pi/4*(x+1))
        jacobian = self.m[comp] * np.pi/4. * (1 + np.tan(np.pi/4.*(x+1.))**2.)
        return 4*np.pi*self.g[comp]/(2*np.pi*hbar)**3 * k**2*np.exp(-np.sqrt(self.m[comp]**2+k**2)/T) * jacobian

    # one-component pressure (antibaryons included here as separate component automatically)
    def pPart(self, comp, T, mu, p, Sigma, K):

        def main_part(comp, x, theta):
            k = self.m[comp] * np.tan(np.pi/4*(x+1))
            return 2.*T*np.cosh(mu/T)*np.exp(-self.v_operator(comp, k, theta)*p/T - self.s_operator(comp, k, theta)*Sigma/T - self.c_operator(comp, k, theta)*K/T)

        def int_func(x):
            return integrate.quadrature(lambda theta: self.momentum_part(comp, x, T)*main_part(comp, x, theta)*np.sin(theta), 0., np.pi/2.)[0]
        return integrate.quad(int_func, -1., 1.)[0]

    # one-component Induced Surface Tension (antibaryons included here as separate component automatically)
    def SigmaPart(self, comp, T, mu, p, Sigma, K):

        def main_part(comp, x, theta):
            k = self.m[comp] * np.tan(np.pi/4*(x+1))
            return 2.*self.alpha_p*T*self.R[comp] * (self.b + (1-self.b)/self.gamma(comp, k)) * np.cosh(mu/T) * np.exp(-self.v_operator(comp, k, theta)*p/T - self.s_operator(comp, k, theta)*self.alpha*Sigma/T - self.c_operator(comp, k, theta)*K/T)

        def int_func(x):
            return integrate.quadrature(lambda theta: self.momentum_part(comp, x, T)*main_part(comp, x, theta)*np.sin(theta), 0., np.pi/2.)[0]

        return integrate.quad(int_func, -1., 1.)[0]

    # one-component Induced Curvature Tension (antibaryons included here as separate component automatically)
    def KPart(self, comp, T, mu, p, Sigma, K):

        def main_part(comp, x, theta):
            k = self.m[comp] * np.tan(np.pi/4*(x+1))
            return 2.*self.beta_p*T*self.R[comp]**2 * (self.b + (1-self.b)/self.gamma(comp, k)) * np.cosh(mu/T) * np.exp(-self.v_operator(comp, k, theta)*p/T - self.s_operator(comp, k, theta)*self.alpha*Sigma/T - self.c_operator(comp, k, theta)*self.beta*K/T)

        def int_func(x):
            return integrate.quadrature(lambda theta: self.momentum_part(comp, x, T)*main_part(comp, x, theta)*np.sin(theta), 0., np.pi/2.)[0]
        return integrate.quad(int_func, -1., 1.)[0]

    def EoS(self, root, T, *mu):
        p, Sigma, K = root
        return (
            p - sum(self.pPart(comp, T, mu[comp], p, Sigma, K) for comp in range(self.num_of_components)),
            Sigma - sum(self.SigmaPart(comp, T, mu[comp], p, Sigma, K) for comp in range(self.num_of_components)),
            K - sum(self.KPart(comp, T, mu[comp], p, Sigma, K) for comp in range(self.num_of_components))
        )

    def root(self, T, *mu, p0=[1., 1., 1.], **kwargs):
        return fsolve(self.EoS, p0, args=(T, *mu), **kwargs)

    def v_eff_ratio(self, T, mu, p, Sigma, K):

        def phi_int(operator):
            def int_func(x): return integrate.quadrature(lambda theta: self.momentum_part(x, T) *
                                                         np.sin(theta)*operator(self.m * np.tan(np.pi/4*(x+1)), theta), 0., np.pi/2.)[0]
            return integrate.quad(int_func, -1., 1.)[0]

        v_numerator = p * phi_int(self.v_operator) + Sigma * phi_int(self.s_operator) + K * phi_int(self.c_operator)
        v_denumerator = p * phi_int(lambda x, y: 1.)
        v_eff = v_numerator / v_denumerator
        return v_eff/(4*self.v)

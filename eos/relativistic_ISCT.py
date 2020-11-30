import numpy as np
import scipy.integrate as integrate
from scipy.optimize import fsolve

from main import Eq_of_state, hbar


class Relativistic_ISCT(Eq_of_state):
    def __init__(self, alpha=1., beta=1., m=940., R=0.5, g=3., alpha_p=1., beta_p=1., a=0.775, b=1., eos=None, components=1):
        self.num_of_components = components
        self.m = m
        self.R = R
        self.g = g
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

    def delta_c1(self, theta):
        return np.cos(theta) - 1

    def delta_c2(self, theta):
        return np.cos(theta)**2 - 1

    def gamma(self, k):
        return np.sqrt(self.m**2 + k**2)/self.m

    def v_operator(self, k, theta):
        return self.v/self.gamma(k)

    def s_operator(self, k, theta):
        return self.s*(1./2./self.gamma(k)*(1. + self.delta_c1(theta)/2.) + self.a/2.*np.abs(np.sin(theta)))

    def c_operator(self, k, theta):
        return self.c/2./self.gamma(k)*(1. + self.delta_c1(theta)/2. + self.delta_c2(theta)/6.)

    def momentum_part(self, x, T):  # after change of variables k -> m tan( pi/4 (x + 1)) and integration limits (0, inf) -> (-1, 1)
        k = self.m * np.tan(np.pi/4*(x+1))
        jacobian = self.m * np.pi/4. * (1 + np.tan(np.pi/4.*(x+1.))**2.)
        return 4*np.pi*self.g/(2*np.pi*hbar)**3 * k**2*np.exp(-np.sqrt(self.m**2+k**2)/T) * jacobian

    def pPart(self, T, mu, p, Sigma, K):

        def main_part(x, theta):
            k = self.m * np.tan(np.pi/4*(x+1))
            return T*np.exp(mu/T - self.v_operator(k, theta)*p/T - self.s_operator(k, theta)*Sigma/T - self.c_operator(k, theta)*K/T)

        def int_func(x): return integrate.quadrature(lambda theta: self.momentum_part(
            x, T)*main_part(x, theta)*np.sin(theta), 0., np.pi/2.)[0]
        return integrate.quad(int_func, -1., 1.)[0]

    def SigmaPart(self, T, mu, p, Sigma, K):

        def main_part(x, theta):
            k = self.m * np.tan(np.pi/4*(x+1))
            return self.alpha_p*T*self.R * (self.b + (1-self.b)/self.gamma(k)) * np.exp(mu/T - self.v_operator(k, theta)*p/T - self.s_operator(k, theta)*self.alpha*Sigma/T - self.c_operator(k, theta)*K/T)

        def int_func(x): return integrate.quadrature(lambda theta: self.momentum_part(
            x, T)*main_part(x, theta)*np.sin(theta), 0., np.pi/2.)[0]
        return integrate.quad(int_func, -1., 1.)[0]

    def KPart(self, T, mu, p, Sigma, K):

        def main_part(x, theta):
            k = self.m * np.tan(np.pi/4*(x+1))
            return self.beta_p*T*self.R**2 * (self.b + (1-self.b)/self.gamma(k)) * np.exp(mu/T - self.v_operator(k, theta)*p/T - self.s_operator(k, theta)*self.alpha*Sigma/T - self.c_operator(k, theta)*self.beta*K/T)

        def int_func(x): return integrate.quadrature(lambda theta: self.momentum_part(
            x, T)*main_part(x, theta)*np.sin(theta), 0., np.pi/2.)[0]
        return integrate.quad(int_func, -1., 1.)[0]

    def EoS(self, root, T, *mu):
        p, Sigma, K = root
        return (p - self.pPart(T, *mu, p, Sigma, K), Sigma - self.SigmaPart(T, *mu, p, Sigma, K), K - self.KPart(T, *mu, p, Sigma, K))

    def root(self, T, *mu, p0=[1., 1., 1.]):
        return fsolve(self.EoS, p0, args=(T, *mu))

    def v_eff_ratio(self, T, mu, p, Sigma, K):

        def phi_int(operator):
            def int_func(x): return integrate.quadrature(lambda theta: self.momentum_part(x, T) *
                                                         np.sin(theta)*operator(self.m * np.tan(np.pi/4*(x+1)), theta), 0., np.pi/2.)[0]
            return integrate.quad(int_func, -1., 1.)[0]

        v_numerator = p * phi_int(self.v_operator) + Sigma * phi_int(self.s_operator) + K * phi_int(self.c_operator)
        v_denumerator = p * phi_int(lambda x, y: 1.)
        v_eff = v_numerator / v_denumerator
        return v_eff/(4*self.v)

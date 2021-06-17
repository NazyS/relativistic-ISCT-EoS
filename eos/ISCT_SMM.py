import numpy as np
from main import Eq_of_state, splined_Eq_of_state, hbar

def kron_delta(k, j):
    if k==j:
        return 1.
    else:
        return 0.

class ISCT_SMM_crit_point(Eq_of_state, splined_Eq_of_state):
    def __init__(self,
        k_indexes=[1, 5, 50, 500],
        g_factors=[1., 9., 90., 900.],
        m=938.,               # MeV
        tau=1.9,
        A=0.5,
        B=0.5,
        alpha=[1.1]*4,
        beta=[1.1]*4,
        chi=2.0/3.,
        po=0.16,              # fm-3
        W0=15.8,              # MeV
        Tcep=15.8,            # MeV
        eps0=16.0,            # MeV
        R=0.39,               # fm
                 ):
        super().__init__()  

        self.chosen_indices = np.array(k_indexes)
        self.g_factors = np.array(g_factors)

        self.num_of_eq_in_eos = 3
        self.num_of_components = len(self.chosen_indices)

        self.m = m
        self.tau = tau
        self.A=A
        self.B=B
        self.alpha=alpha
        self.beta=beta
        self.chi = chi
        self.po = po
        self.W0 = W0
        self.Tcep = Tcep
        self.eps0 = eps0
        self.R = R

        self.V1 = 1. / self.po

        self.radiuses = (3.*self.chosen_indices*self.V1/(16.*np.pi))**(1./3.)

    def W(self, T):
        return  self.W0 + T**2 / self.eps0

    def gamma(self, T):
        return T*(self.m*T/(2*np.pi*hbar**2))**1.5

    def p_liquid(self, T, mu):
        return (mu + self.W(T)) / self.V1

    def b_coefficient(self, index, T):
        if index == 1:
            return 4.*np.exp(-self.W(T)/T)
        else:
            return 1.

    # sigma_0 = -sigma_1 and c_0 = - c_1 in the critical point
    # which is directly defined here
    def pPart(self, comp, root, T, *mu):
        p, sigma1, c1 = root
        return self.g_factors[comp]*self.gamma(T)*self.b_coefficient(self.chosen_indices[comp], T)/self.chosen_indices[comp]**self.tau*np.exp((self.p_liquid(T, mu[comp]) - p)*self.V1*self.chosen_indices[comp]/T)

    def sigma1Part(self, comp, root, T, *mu):
        p, sigma1, c1 = root
        return 3.*self.A*self.V1*self.g_factors[comp]*self.gamma(T)*self.b_coefficient(self.chosen_indices[comp], T)/self.chosen_indices[comp]**(self.tau - 1./3.)*np.exp((self.p_liquid(T, mu[comp]) - p)*self.V1*self.chosen_indices[comp]/T - (self.alpha[comp] - 1.)*sigma1*self.chosen_indices[comp]**(2./3.)/T)

    def c1Part(self, comp, root, T, *mu):
        p, sigma1, c1 = root
        return 3.*self.B*self.V1*self.g_factors[comp]*self.gamma(T)*self.b_coefficient(self.chosen_indices[comp], T)/self.chosen_indices[comp]**(self.tau - 2./3.)*np.exp((self.p_liquid(T, mu[comp]) - p)*self.V1*self.chosen_indices[comp]/T - (self.alpha[comp] - 1.)*sigma1*self.chosen_indices[comp]**(2./3.)/T - (self.beta[comp] - 1.)*c1*self.chosen_indices[comp]**(1./3.)/T)

    def EoS(self, root, T, *mu):
        p, sigma1, c1 = root
        return (
            p - sum(self.pPart(comp, root, T, *mu) for comp in range(self.num_of_components)),
            sigma1 - sum(self.sigma1Part(comp, root, T, *mu) for comp in range(self.num_of_components)),
            c1 - sum(self.c1Part(comp, root, T, *mu) for comp in range(self.num_of_components))
        )

    # derived from full non-limited EoS (probably inapplicable here)
    def density_analytical_comp(self, comp, T, *mu, **kwargs):
        root = self.p_eq(T, *mu, format='full', **kwargs)
        # k_index = self.chosen_indices[comp]
        deriv_sigma0_by_mu = -1.
        deriv_c0_by_mu = -1.

        a = 1 + sum(self.pPart(k, root, T, *mu)*self.V1*self.chosen_indices[k]/T for k in range(self.num_of_components))
        b = 1 + sum(self.sigma1Part(k, root, T, *mu)*self.alpha[k]*self.chosen_indices[k]**(2./3.)/T for k in range(self.num_of_components))
        c = 1 + sum(self.c1Part(k, root, T, *mu)*self.beta[k]*self.chosen_indices[k]**(1./3.)/T for k in range(self.num_of_components))

        # delta11 = root[0]*k_index/T
        delta11 = sum(self.pPart(k, root, T, *mu)/T*(self.chosen_indices[k]*kron_delta(k, comp) - deriv_sigma0_by_mu*self.chosen_indices[k]**(2./3.) - deriv_c0_by_mu*self.chosen_indices[k]**(1./3.)) for k in range(self.num_of_components))
        delta12 = sum(self.pPart(k, root, T, *mu)*self.chosen_indices[k]**(2./3.)/T for k in range(self.num_of_components))
        delta13 = sum(self.pPart(k, root, T, *mu)*self.chosen_indices[k]**(1./3.)/T for k in range(self.num_of_components))

        # delta21 = root[1]*k_index/T
        delta21 = sum(self.sigma1Part(k, root, T, *mu)/T*(self.chosen_indices[k]*kron_delta(k, comp) - deriv_sigma0_by_mu*self.chosen_indices[k]**(2./3.) - deriv_c0_by_mu*self.chosen_indices[k]**(1./3.)) for k in range(self.num_of_components))
        delta23 = sum(self.sigma1Part(k, root, T, *mu)*self.chosen_indices[k]**(1./3.)/T for k in range(self.num_of_components))

        # delta31 = root[2]*k_index/T
        delta31 = sum(self.c1Part(k, root, T, *mu)/T*(self.chosen_indices[k]*kron_delta(k, comp) - deriv_sigma0_by_mu*self.chosen_indices[k]**(2./3.) - deriv_c0_by_mu*self.chosen_indices[k]**(1./3.)) for k in range(self.num_of_components))
        delta32 = sum(self.c1Part(k, root, T, *mu)*self.alpha[k]*self.chosen_indices[k]**(2./3.)/T for k in range(self.num_of_components))
        delta21_til = sum(self.sigma1Part(k, root, T, *mu)*self.chosen_indices[k]*self.V1/T for k in range(self.num_of_components))
        delta31_til = sum(self.c1Part(k, root, T, *mu)*self.chosen_indices[k]*self.V1/T for k in range(self.num_of_components))

        num = b*c*delta11 - delta11*delta32*delta23 - c*delta12*delta21 - b*delta13*delta31 + delta12*delta23*delta31 + delta13*delta32*delta21

        denum = a*b*c - a*delta32*delta23 - c*delta12*delta21_til - b*delta13*delta31_til + delta31_til*delta12*delta23 + delta13*delta32*delta21_til

        dictionary = {
            'a':[a],
            'b':[b],
            'c':[c],
            'delta11':[delta11],
            'delta12':[delta12],
            'delta13':[delta13],
            'delta21':[delta21],
            'delta23':[delta23],
            'delta31':[delta31],
            'delta32':[delta32],
            'delta21_til':[delta21_til],
            'delta31_til':[delta31_til],
        }

        return num/denum, dictionary

    def density_analytical(self, T, *mu, **kwargs):
        return sum(self.density_analytical_comp(comp, T, *mu, **kwargs)[0] for comp in range(self.num_of_components))

    def density_analytical_simpl_comp(self, comp, T, *mu, **kwargs):
        root = self.p_eq(T, *mu, format='full', **kwargs)

        num = self.pPart(comp, root, T, *mu)*self.chosen_indices[comp]/T
        denum = 1 + sum(self.pPart(k, root, T, *mu)*self.V1*self.chosen_indices[k]/T for k in range(self.num_of_components))
        return num/denum

    def density_analytical_simpl(self, T, *mu, **kwargs):
        return sum(self.density_analytical_simpl_comp(comp, T, *mu, **kwargs) for comp in range(self.num_of_components))
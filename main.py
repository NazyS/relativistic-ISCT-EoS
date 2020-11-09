import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve
from scipy.misc import derivative

# for storing func results in cache for multiple use
from functools import lru_cache


def partial_derivative(func, var=0, point=[], dx=1., order=3, **kwargs):
    args = point[:]

    def wraps(x):
        args[var] = x
        return func(*args, **kwargs)
    return derivative(wraps, point[var], dx=dx, order=order)


hbar = 197.326


@lru_cache(maxsize=32)
def phi(T, m):
    def func(p):
        return p**2 * np.exp(-np.sqrt(p**2+m**2)/T)

    integration = quad(func, 0, np.inf)[0]
    return (4*np.pi)/(2*np.pi*hbar)**3 * integration


# # Main equation of state Class

class Eq_of_state:

    # def __init__(self):
    #     self.num_of_eq_in_eos = num_of_eq_in_eos

    @lru_cache(maxsize=512)
    def p_eq(self, T, *mu, init=None, format='p only', **kwargs):
        init = init if init else [0.]*self.num_of_eq_in_eos
        root = fsolve(self.EoS, init, args=(T, *mu),
                      maxfev=10000,
                      )
        diff = self.EoS(root, T, *mu)
        if sum(diff[i]**2/root[i]**2 for i in range(len(diff))) > 1e-4:
            print('error for T={}, mu={}'.format(T, mu[0]))
            return np.nan if format == 'p only' else np.array([np.nan]*len(init))
        # print(root)
        return root[0] if format == 'p only' else root

    #     entropy density
    def entropy(self, T, *mu, **kwargs):
        return partial_derivative(self.p_eq, 0, [T, *mu], **kwargs)

    # particle density of single component
    @lru_cache(maxsize=512)
    def density_comp(self, comp_number, T, *mu, **kwargs):
        return partial_derivative(self.p_eq, 1+comp_number, [T, *mu], **kwargs)

    # baryon minus its imaginary antibaryon density for each component
    # 0.5 factor due to same input from actual antibaryon term with opposite sign in total baryon charge density
    def baryon_minus_antibaryon_comp_density_halfed(self, comp_number, T, *mu, **kwargs):
        mu_inverted = tuple([-el for el in mu])
        return 0.5*(self.density_comp(comp_number, T, *mu, **kwargs) - self.density_comp(comp_number, T, *mu_inverted, **kwargs))

    # baryon charge density which is baryon density minus antibaryon charge density
    def density_baryon(self, T, *mu, **kwargs):
        return sum(np.sign(mu)[i] * self.baryon_minus_antibaryon_comp_density_halfed(i, T, *mu, **kwargs) for i in range(self.num_of_components))
        # return self.density_comp(1, T, *mu, **kwargs)

    # total particle density
    # def density(self, T, *mu, **kwargs):
    #     return sum(self.density_comp(i, T, *mu, **kwargs) for i in range(self.num_of_components))

    def energy(self, T, *mu, **kwargs):
        # return T*self.entropy(T, *mu) + sum( mu[i]*self.density_comp(i, T, *mu) for i in range(self.num_of_components) ) - self.p_eq(T,*mu)  # for multi mu_k case
        return T*self.entropy(T, *mu, **kwargs) + sum(mu[i]*self.baryon_minus_antibaryon_comp_density_halfed(i, T, *mu, **kwargs) for i in range(self.num_of_components)) - self.p_eq(T, *mu, **kwargs)

    # entropy by baryon density ratio ( \sigma  = s / n_B )
    def entr_by_density_baryon(self, T, *mu, **kwargs):
        return self.entropy(T, *mu, **kwargs)/self.density_baryon(T, *mu, **kwargs)

    def speed_of_s_sq(self, T, *mu, **kwargs):

        def d_mu_by_d_T(T, *mu):
            num = partial_derivative(
                self.entr_by_density_baryon, 0, [T, *mu], **kwargs)
            denum = partial_derivative(
                self.entr_by_density_baryon, 1, [T, *mu], **kwargs)
            # dealing with infinite sigma if \mu = 0 hence baryonic density = 0
            # might be dangerous
            if np.isnan(num/denum):
                return 0.
            else:
                return num/denum

        d_mu_d_T = d_mu_by_d_T(T, *mu)

        num = partial_derivative(self.p_eq, 0, [T, *mu], **kwargs) - partial_derivative(
            self.p_eq, 1, [T, *mu], **kwargs)*d_mu_d_T
        denum = partial_derivative(self.energy, 0, [T, *mu], **kwargs) - partial_derivative(self.energy, 1, [
            T, *mu], **kwargs)*d_mu_d_T
        return num/denum

    # speed of sound for multiple chemical potentials
    def speed_of_s_sq_MULT(self, T, *mu, **kwargs):

        def deriv_by_sigma_with_const_T(func, T, *mu):
            num = sum(partial_derivative(
                func, comp+1, [T, *mu], **kwargs) for comp in range(self.num_of_components))
            denum = sum(partial_derivative(self.entr_by_density_baryon, comp+1,
                                           [T, *mu], **kwargs) for comp in range(self.num_of_components))
            # dealing with infinite sigma if \mu = 0 hence baryonic density = 0
            # might be dangerous
            if np.isnan(num/denum):
                return 0.
            else:
                return num/denum

        def sigma_deriv_by_T(T, *mu):
            val = partial_derivative(
                self.entr_by_density_baryon, 0, [T, *mu], **kwargs)
            # dealing with infinite sigma if \mu = 0 hence baryonic density = 0
            # might be dangerous
            if np.isnan(val):
                return 0.
            else:
                return val

        sigma_deriv = sigma_deriv_by_T(T, *mu)

        num = partial_derivative(self.p_eq, 0, [
                                 T, *mu], **kwargs) - deriv_by_sigma_with_const_T(self.p_eq, T, *mu)*sigma_deriv
        denum = partial_derivative(self.energy, 0, [
                                   T, *mu], **kwargs) - deriv_by_sigma_with_const_T(self.energy, T, *mu)*sigma_deriv
        return num/denum


# --------------------------------------------------------------------------------------------------------------------------


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


# EoS with modified function for excluded volume f(p) subclass
# Usable for multicomponent case

class modified_exc_volume(Eq_of_state):
    '''
    input list of size eq. to num. of comp. for each parameter

    EoS with modified function for excluded volume f(p)
    Usable for multicomponent case
    '''

    def __init__(self, q1=[0.5], a1=[0.06], q2=[0.25], a2=[0.01], q3=[0.], a3=[0.], p0=[20.],
                 R=[0.4], m=[940.], g=[1.]
                 ):
        self.m = m
        self.num_of_components = len(m)
        # super().__init__(self.EoS)

        # f(p) function parameters
        self.q1 = q1 if len(
            q1) == self.num_of_components else q1*self.num_of_components
        self.a1 = a1 if len(
            a1) == self.num_of_components else a1*self.num_of_components
        self.q2 = q2 if len(
            q2) == self.num_of_components else q2*self.num_of_components
        self.a2 = a2 if len(
            a2) == self.num_of_components else a2*self.num_of_components

        self.q3 = q3 if len(
            q3) == self.num_of_components else q3*self.num_of_components
        self.a3 = a3 if len(
            a3) == self.num_of_components else a3*self.num_of_components

        self.p0 = p0 if len(
            p0) == self.num_of_components else p0*self.num_of_components

        # hard-core radii
        self.R = R if len(R) == self.num_of_components else R * \
            self.num_of_components

        # degeneracy factor
        self.g = g if len(g) == self.num_of_components else g * \
            self.num_of_components

    num_of_eq_in_eos = 1

    def f(self, p, componet):
        return 1 / (1 + self.a1[componet]*(p/self.p0[componet])**self.q1[componet] + self.a2[componet]*(p/self.p0[componet])**self.q2[componet] + self.a3[componet]*(p/self.p0[componet])**self.q3[componet])

    def b(self, componet):                                                      # Excluded volume b
        return 4 * self.R[componet]**3 * 4*np.pi/3

    # multicomponent Equation of state
    def EoS(self, p, T, *mu):
        return sum(T*self.g[componet]*phi(T, self.m[componet]) * np.exp((mu[componet] - self.b(componet)*self.f(p, componet)*p)/T) - p for componet in range(self.num_of_components))

    # generates a string from values of array with corresponding names
    def gen_str_for_array(self, array, name='name', quantity=''):

        # efficient function to check if all values in array are same
        def check_if_equal(array):
            return len(set(array)) == 1

        # if quantity is not empty, then add space in front for good visual separation
        if quantity:
            quantity = ' ' + quantity

        # if all values in array are same then display its single entry for all components
        if check_if_equal(array):
            string = '${}={:.2f}${}'.format(name, array[0], quantity)
        # else display entry for each one separately
        else:
            string = ''
            i = 1
            for elem in array:
                if i != 1:
                    string += ', '
                string += '${}_{:d}='.format(name, i) + \
                    '{:.2f}${}'.format(elem, quantity)
                i += 1
        return string

    def parameters_string(self, format='short'):
        if format == 'full':
            string = self.gen_str_for_array(
                self.m, name='m', quantity='MeV') + ', ' + self.gen_str_for_array(self.R, name='R', quantity='fm')
            string += ',\n' + self.gen_str_for_array(self.q1, name='q_1', quantity='') + \
                ', ' + self.gen_str_for_array(self.a1, name='a_1', quantity='')
            string += ', ' + self.gen_str_for_array(self.q2, name='q_2', quantity='') + \
                ', ' + self.gen_str_for_array(self.a2, name='a_2', quantity='')
            string += ',\n' + \
                self.gen_str_for_array(
                    self.p0, name='p_0', quantity='$MeV/fm^{3}$')
            return string
        elif format == 'short':
            string = self.gen_str_for_array(
                self.m, name='m', quantity='MeV') + ',\n' + self.gen_str_for_array(self.q1, name='q_1', quantity='')
            string += ', ' + \
                self.gen_str_for_array(self.a1, name='a_1', quantity='')
            string += ', ' + self.gen_str_for_array(self.q2, name='q_2', quantity='') + \
                ', ' + self.gen_str_for_array(self.a2, name='a_2', quantity='')
            return string
        elif format == 'extra':
            string = self.gen_str_for_array(
                self.q3, name='q_3', quantity='') + ', ' + self.gen_str_for_array(self.a3, name='a_3', quantity='')
            return string
        else:
            print('Wrong format!')


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


# # ISCT equation of state Subclass

class ISCT(Eq_of_state):

    '''
    input list of size eq. to num. of comp. for each parameter

    ISCT with modified function for excluded volume f(p)
    Usable for multicomponent case
    '''

    def __init__(self,
                 q1=0.5, q2=0.25, a1=1., a2=1., a3=1.,
                 p_max=150., delta_p=1.,  p0=20.,
                 # parameters for asymptotic f (vdw case) on infinite p   (used in spline interp)
                 ad=None, qd=0.5, ad_factor=1.,
                 # parameters for f spline polynom for connecting f_with_max and f_limit and limits of spline interval
                 a_spline=None, p_A=np.inf, p_B=0.,
                 R=[0.4], m=[20.], g=[4.],
                 Alpha=[1.25], Beta=[2.45], A=[0.5], B=[0.5],
                 ):
        self.m = m
        self.num_of_components = len(m)
        # super().__init__(self.EoS)

        # f(p) function parameters
        self.num_of_powers = 2

        self.q_parameters = [q1, q2]
        self.a_parameters = [a1, a2, a3]

        # new f(p) function parameters
        self.q1 = q1
        self.q2 = q2
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.p_max = p_max
        self.delta_p = delta_p

        self.p0 = p0

        # parameters for f spline
        self.a_spline = a_spline
        self.p_A = p_A
        self.p_B = p_B

        # hard-core radii
        self.R = R if len(R) == self.num_of_components else R * \
            self.num_of_components
        # degeneracy factor
        self.g = g if len(g) == self.num_of_components else g * \
            self.num_of_components

        # asymptotic f func parameters
        self.ad_factor = ad_factor
        self.ad = ad if ad else self.ad_factor * \
            np.sqrt(self.p0*self.b(0)/8./self.m[0])
        self.qd = qd

        # ISCT parameters
        self.Alpha = Alpha if len(
            Alpha) == self.num_of_components else Alpha*self.num_of_components
        self.Beta = Beta if len(
            Beta) == self.num_of_components else Beta*self.num_of_components
        self.A = A if len(A) == self.num_of_components else A * \
            self.num_of_components
        self.B = B if len(B) == self.num_of_components else B * \
            self.num_of_components

    def b(self, component):
        return self.v(component)*4

    def f(self, p, componet):
        return 1 / (1 + sum(self.a_parameters[elem]*(p / self.p0) ** self.q_parameters[elem] for elem in range(self.num_of_powers)))

    def f_with_max(self, p, componet):
        return 1 / (1 + self.a1*(p/self.p0)**self.q1 + (p/self.p0)**self.q2*(self.a2 + self.a3/((p-self.p_max)**2 + self.delta_p**2)))

    def f_limit(self, p, componet):
        return 1/(1 + self.ad*(p/self.p0)**self.qd)

    def f_spline(self, p, a_spline, componet):
        return np.polynomial.polynomial.polyval(p, a_spline)

    def f_stitched(self, p, a_spline, p_A, p_B, componet):
        if p < p_A:
            return self.f_with_max(p, componet)
        elif p < p_B:
            return self.f_spline(p, a_spline, componet)
        else:
            return self.f_limit(p, componet)

    num_of_eq_in_eos = 3

    # Thermal densities
    @lru_cache(maxsize=128)
    def phi(self, T, component):
        return quad(lambda p: self.g[component] * p**2 / (2 * np.pi**2 * hbar**3) * np.exp(-np.sqrt(p**2 + self.m[component]**2) / T), 0, np.inf)[0]

    # Values for eigensurface s, eigenvolume v and eigencurvature c for each component

    def s(self, component):
        return 4 * np.pi * self.R[component]**2

    def v(self, component):
        return 4 / 3 * np.pi * self.R[component]**3

    def c(self, component):
        return self.s(component) / self.R[component]

    # Partial pressure p_k, Ind Surf Tension Sigma_k and Ind Curv Tension K_k for each component

    def pPart(self, p, Sigma, K, T, mu, component):
        return T*self.phi(T, component) * np.exp((mu/T - self.v(component)*p/T - self.s(component)*Sigma/T - self.c(component)*K/T)*self.f_stitched(p, self.a_spline, self.p_A, self.p_B, component))

    def SigmaPart(self, p, Sigma, K, T, mu, component):
        return self.A[component] * self.R[component] * T*self.phi(T, component) * np.exp((mu/T - self.v(component)*p/T - self.Alpha[component]*self.s(component)*Sigma/T - self.c(component)*K/T)*self.f_stitched(p, self.a_spline, self.p_A, self.p_B, component))

    def KPart(self, p, Sigma, K, T, mu, component):
        return self.B[component] * self.R[component]**2 * T*self.phi(T, component) * np.exp((mu/T - self.v(component)*p/T - self.Alpha[component]*self.s(component)*Sigma/T - self.Beta[component]*self.c(component)*K/T)*self.f_stitched(p, self.a_spline, self.p_A, self.p_B, component))

    # Full pressure p,  IST Sigma and ICT K  for system
    def pFull(self, p, Sigma, K, T, *mu):
        return sum(self.pPart(p, Sigma, K, T, mu[component], component) for component in range(self.num_of_components))

    def SigmaFull(self, p, Sigma, K, T, *mu):
        return sum(self.SigmaPart(p, Sigma, K, T, mu[component], component) for component in range(self.num_of_components))

    def KFull(self, p, Sigma, K, T, *mu):
        return sum(self.KPart(p, Sigma, K, T, mu[component], component) for component in range(self.num_of_components))

    def EoS(self, root, T, *mu):
        p, Sigma, K = root
        return (self.pFull(p, Sigma, K, T, *mu) - p, self.SigmaFull(p, Sigma, K, T, *mu) - Sigma, self.KFull(p, Sigma, K, T, *mu) - K)

    # Generating string of EoS parameters
    # generates a string from values of array with corresponding names
    def gen_str_for_array(self, array, name='name', quantity=''):

        # efficient function to check if all values in array are same
        def check_if_equal(array):
            return len(set(array)) == 1

        # if quantity is not empty, then add space in front for good visual separation
        if quantity:
            quantity = ' ' + quantity

        # if all values in array are same then display its single entry for all components
        if check_if_equal(array):
            string = '${}={:.2f}${}'.format(name, array[0], quantity)
        # else display entry for each one separately
        else:
            string = ''
            i = 1
            for elem in array:
                if i != 1:
                    string += ', '
                string += '${}_{:d}='.format(name, i) + \
                    '{:.2f}${}'.format(elem, quantity)
                i += 1
        return string

    def parameters_string(self, format='short'):
        if format == 'basic':
            string = self.gen_str_for_array(
                self.m, name='m', quantity='MeV') + ', ' + self.gen_str_for_array(self.R, name='R', quantity='fm')
            string += ', ' + \
                self.gen_str_for_array(self.g, name='g', quantity='')
            return string

        if format == 'f parameters':
            string = self.gen_str_for_array(self.q_parameters, name='q', quantity='') + \
                ',\n' + \
                self.gen_str_for_array(
                    self.a_parameters, name='a', quantity='')
            string += ',\n' + self.gen_str_for_array([self.delta_p], name='\\delta p', quantity='') + ', ' + self.gen_str_for_array(
                [self.p_max], name='p_m', quantity='') + ', ' + self.gen_str_for_array([self.p0], name='p_0', quantity='$MeV/fm^{3}$')
            return string

        elif format == 'eos parameters':
            string = self.gen_str_for_array(self.Alpha, name='\\alpha', quantity='') + \
                ', ' + \
                self.gen_str_for_array(
                    self.Beta, name='\\beta', quantity='') + ',\n'
            string += self.gen_str_for_array(self.A, name='A', quantity='') + \
                ', ' + self.gen_str_for_array(self.B, name='B', quantity='')
            return string
        else:
            print('Wrong format!')


# # # ISCT with SMM equation of state Subclass

# class ISCT_with_SMM(Eq_of_state):

#     numM = 1

#     num_of_eq_in_eos = 3

#     # EoS parameters

#     Alpha = [1.25]
#     Beta = [2.45]
#     A = 0.5
#     B = 0.5
#     ka = 3.
#     g = [4.] * numM

#     # (*Masses calculated from  SMM (PHYSICAL REVIEW C,VOLUME 62,044320) *)
#     mNucl = 938.9              #Mass of nucleon (in MeV)
#     W0 = 16.                    #the volume binding energy per nucleon (in MeV)
#     eps0 = 16.                  #contribution of the excited states taken in the Fermi-gas approximation (in MeV)
#     def W(self, T):                  #General binding energy
#         return -self.W0 - T**2/self.eps0
#     def m(self, T, component):
#         return (k+1)*(self.mNucl + self.W(T))

#     # Radiuses calculated using normal nuclear density
#     rho0 = 0.16                                                    #Normal nuclear density
#     def R(self,k):
#         return (3/(4*np.pi)*(k+1)/self.rho0)**(1/3)  #have to k+1 because numerations start from 0
#     def L(self, k):
#         return self.ka/3 * self.R(component)

#     # (*Chemical potentials*)
#     aparam = -4.414*10**(-4)
#     def Mu(self, k, MU):
#         return (k+1)*(MU + self.aparam*MU**3)


#     # Thermal densities
#     @lru_cache(maxsize=32)
#     def phi(self, T, component):
#         return   quad( lambda p: self.g[component] * p**2 / (2 * np.pi**2 * 197.327**3) *np.exp( -np.sqrt( p**2 + self.m(k,T)**2) / T), 0, np.inf)[0]


#     # Values for eigensurface s, eigenvolume v and eigencurvature c for each component
#     def s(self, k):
#         return 4 * np.pi * self.L(component)**2
#     def v(self,k):
#         return 4 / 3 * np.pi * self.L(component)**3
#     def c(self, k):
#         return self.s(component) / self.L(component)


#     # Partial pressure p_k, Ind Surf Tension Sigma_k and Ind Curv Tension K_k for each component
#     def pPart(self, p, Sigma, K, T, mu, component):
#         return T*self.phi(T, component) * np.exp( mu/T - self.v(component)*p/T - self.s(component)*Sigma/T - self.c(component)*K/T )
#     def SigmaPart(self, p, Sigma, K, T, mu, component):
#         return self.A * self.L(component) * T*self.phi(T, component) * np.exp( mu/T - self.v(component)*p/T - self.Alpha[component]*self.s(component)*Sigma/T - self.c(component)*K/T )
#     def KPart(self, p, Sigma, K, T, mu, component):
#         return self.B * self.L(component)**2 * T*self.phi(T, component) * np.exp( mu/T - self.v(component)*p/T - self.Alpha[component]*self.s(component)*Sigma/T - self.Beta[component]*self.c(component)*K/T )

#     # Full pressure p,  IST Sigma and ICT K  for system
#     def pFull(self, T, mu, root):
#         return sum( self.pPart(p, Sigma, K, T, mu, component) for k in range(self.numM) )
#     def SigmaFull(self, T, mu, root):
#         return sum( self.SigmaPart(p, Sigma, K, T, mu, component) for k in range(self.numM) )
#     def KFull(self, T, mu, root):
#         return sum( self.KPart(p, Sigma, K, T, mu, component) for k in range(self.numM) )

#     def EoS(self, args, T, MU):
#         p, Sigma, K = args
#         return ( self.pFull(T, mu, root) - p, self.SigmaFull(T, mu, root) - Sigma, self.KFull(T, mu, root) - K )


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


# # Nucleon gas: quantum

class Nucl_gas_quant(Eq_of_state):

    b = 1.
    m = 938.
    g = 4.

    num_of_eq_in_eos = 1

    def EoS(self, p, T, mu):
        def E(k):
            return np.sqrt(k**2 + self.m**2)

        def integr_func(k): return k**2 / \
            (np.exp((-mu + E(k) + self.m*self.b*p/E(k))/T) + 1)

        return self.g*T*(4*np.pi)/(2*np.pi*hbar)**3 * quad(integr_func, 0, np.inf, )[0] - p


# # Nucleon + Antinucleon gas: quantum

class Nucl_Antinucl(Eq_of_state):

    b = 1.
    m = 938.
    g = 4.

    num_of_eq_in_eos = 1

    def EoS(self, p, T, mu):
        def E(k):
            return np.sqrt(k**2 + self.m**2)

        def integr_func(k): return k**2 * (1/(np.exp((-mu + E(k) + self.m*self.b *
                                                      p/E(k))/T) + 1) + 1/(np.exp((mu + E(k) + self.m*self.b*p/E(k))/T) + 1))

        return self.g*T*(4*np.pi)/(2*np.pi*hbar)**3 * quad(integr_func, 0, np.inf, )[0] - p


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

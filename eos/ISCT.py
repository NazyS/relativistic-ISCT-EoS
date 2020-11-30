import numpy as np
from scipy.integrate import quad

from main import Eq_of_state, hbar

# for storing func results in cache for multiple use
from functools import lru_cache

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

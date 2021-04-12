import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve, newton
from scipy.misc import derivative

# for storing func results in cache for multiple use
from functools import lru_cache


def partial_derivative(func, var=0, point=[], dx=1e-1, order=3, n=1, **kwargs):
    args = point[:]

    def wraps(x):
        args[var] = x
        return func(*args, **kwargs)
    return derivative(wraps, point[var], dx=dx, order=order+(n-1)*2, n=n)


hbar = 197.3269804

# thermal density function
@lru_cache(maxsize=32)
def phi(T, m):
    def func(p):
        return p**2 * np.exp(-np.sqrt(p**2+m**2)/T)
    integration = quad(func, 0, np.inf)[0]
    return (4*np.pi)/(2*np.pi*hbar)**3 * integration


# Main equation of state Class
class Eq_of_state:
    '''
    define "num_of_eq_in_eos" and "num_of_components" variables while inheriting
    define "EoS(root, T, *mu)" method or overwrite "p_eq(T, *mu)" method

    Main class for equation of state and speed of sound calculations
    '''

    @lru_cache(maxsize=512)
    def p_eq(self, T, *mu, init=None, format='p only', method='fsolve', **kwargs):
        init = init if init else np.full(self.num_of_eq_in_eos, 1.)
        if method=='fsolve':
            root = fsolve(self.EoS, init, args=(T, *mu),
                        maxfev=10000
                        )
        elif method=='newton':
            root = newton(self.EoS, init, args=(T, *mu), maxiter=10000)
        else:
            raise Exception('Wrong method specified!')

        diff = self.EoS(root, T, *mu)
        if sum(diff[i]**2/root[i]**2 for i in range(len(diff))) > 1e-4:
            print('error for T={}, mu={}'.format(T, mu[0]))
            return np.nan if format == 'p only' else np.full(self.num_of_eq_in_eos, np.nan)
        return root[0] if format == 'p only' else root

    # entropy density
    @lru_cache(maxsize=512)
    def entropy(self, T, *mu, **kwargs):
        return partial_derivative(self.p_eq, 0, [T, *mu], **kwargs)

    # baryon density of a single component
    @lru_cache(maxsize=512)
    def density_baryon_comp(self, comp, T, *mu, **kwargs):
        return partial_derivative(self.p_eq, 1+comp, [T, *mu], **kwargs)

    def density_baryon(self, T, *mu, **kwargs):
        return sum(self.density_baryon_comp(i, T, *mu, **kwargs) for i in range(len(mu)))

    def energy(self, T, *mu, **kwargs):
        return T*self.entropy(T, *mu, **kwargs) + sum(mu[i]*self.density_baryon_comp(i, T, *mu, **kwargs) for i in range(len(mu))) - self.p_eq(T, *mu, **kwargs)

    # entropy by baryon density ratio ( \sigma  = s / n_B )
    def sigma(self, T, *mu, **kwargs):
        return self.entropy(T, *mu, **kwargs)/self.density_baryon(T, *mu, **kwargs)

    def speed_of_s_sq(self, T, *mu, warning=False, **kwargs):

        def d_mu_by_d_T(T, *mu):
            num = partial_derivative(self.sigma, 0, [T, *mu], **kwargs)
            denum = partial_derivative(self.sigma, 1, [T, *mu], **kwargs)
            res = -num/denum

            # dealing with infinite sigma if \mu = 0 hence baryonic density = 0 and constant
            # might be dangerous
            if np.isnan(res):
                if warning:
                    print('nan value warning for T={}'.format(T))
                return 0.
            else:
                return res

        d_mu_d_T = d_mu_by_d_T(T, *mu)

        num = self.entropy(T, *mu, **kwargs) + self.density_baryon(T, *mu, **kwargs)*d_mu_d_T
        denum = partial_derivative(self.energy, 0, [T, *mu], **kwargs) + \
            partial_derivative(self.energy, 1, [T, *mu], **kwargs)*d_mu_d_T
        return num/denum

    # speed of sound for multiple chemical potentials
    # NOT TESTED
    def speed_of_s_sq_MULT(self, T, *mu, warning=False, **kwargs):

        def deriv_by_sigma_with_const_T(func, T, *mu):
            num = sum(partial_derivative(func, comp+1, [T, *mu], **kwargs) for comp in range(self.num_of_components))
            denum = sum(partial_derivative(self.sigma, comp+1,
                                           [T, *mu], **kwargs) for comp in range(self.num_of_components))
            res = num/denum

            # dealing with infinite sigma if \mu = 0 hence baryonic density = 0 and constant
            # might be dangerous
            if np.isnan(res):
                if warning:
                    print('nan value warning for T={}'.format(T))
                return 0.
            else:
                return res

        def sigma_deriv_by_T(T, *mu):
            val = partial_derivative(self.sigma, 0, [T, *mu], **kwargs)
            # dealing with infinite sigma if \mu = 0 hence baryonic density = 0 and constant
            # might be dangerous
            if np.isnan(val):
                if warning:
                    print('nan value warning for T={}'.format(T))
                return 0.
            else:
                return val

        sigma_deriv = sigma_deriv_by_T(T, *mu)

        num = partial_derivative(self.p_eq, 0, [T, *mu], **kwargs) - \
            deriv_by_sigma_with_const_T(self.p_eq, T, *mu)*sigma_deriv
        denum = partial_derivative(self.energy, 0, [T, *mu], **kwargs) - \
            deriv_by_sigma_with_const_T(self.energy, T, *mu)*sigma_deriv
        return num/denum

    @lru_cache(maxsize=512)
    def cumulant_per_vol(self, order, comp, T, *mu, **kwargs):
        derivative = partial_derivative(self.p_eq, 1+comp, [T, *mu], n=order, **kwargs)
        return T**(order - 1)*derivative

    def cumul_lin_ratio(self, T, *mu, **kwargs):
        return T*self.cumulant_per_vol(1, 0, T, *mu, **kwargs)/mu[0]/self.cumulant_per_vol(2, 0, T, *mu, **kwargs)

    def cumul_sq_ratio(self, T, *mu, **kwargs):
        return 1. - self.cumulant_per_vol(3, 0, T, *mu, **kwargs)*self.cumulant_per_vol(1, 0, T, *mu, **kwargs)/self.cumulant_per_vol(2, 0, T, *mu, **kwargs)**2


def get_changable_dx(var):
    # suitable for T or mu (prob)
    if var<100.:
        return 1e-3
    elif var<500.:
        return 1e-1
    else:
        return 1.
import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve
from scipy.misc import derivative

# for storing func results in cache for multiple use
from functools import lru_cache


def partial_derivative(func, var=0, point=[], dx=1e-1, order=3, **kwargs):
    args = point[:]

    def wraps(x):
        args[var] = x
        return func(*args, **kwargs)
    return derivative(wraps, point[var], dx=dx, order=order)


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

    @lru_cache(maxsize=512)
    def p_eq(self, T, *mu, init=None, format='p only', **kwargs):
        init = init if init else np.full(self.num_of_eq_in_eos, 0.)
        root = fsolve(self.EoS, init, args=(T, *mu),
                      maxfev=10000
                      )
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

    # particle density of single component
    # @lru_cache(maxsize=512)
    # def density_comp(self, comp_number, T, *mu, **kwargs):
    #     return partial_derivative(self.p_eq, 1+comp_number, [T, *mu], **kwargs)

    # baryon minus its imaginary antibaryon density for each component
    # 0.5 factor due to same input from actual antibaryon term with opposite sign in total baryon charge density
    # def baryon_minus_antibaryon_comp_density_halfed(self, comp_number, T, *mu, **kwargs):
    #     mu_inverted = tuple([-el for el in mu])
    #     return 0.5*(self.density_comp(comp_number, T, *mu, **kwargs) - self.density_comp(comp_number, T, *mu_inverted, **kwargs))

    # baryon charge density which is baryon density minus antibaryon charge density
    # def density_baryon(self, T, *mu, **kwargs):
    #     return sum(np.sign(mu)[i] * self.baryon_minus_antibaryon_comp_density_halfed(i, T, *mu, **kwargs) for i in range(self.num_of_components))

    # total particle density
    # def density(self, T, *mu, **kwargs):
    #     return sum(self.density_comp(i, T, *mu, **kwargs) for i in range(self.num_of_components))

    # def energy(self, T, *mu, **kwargs):
    #     return T*self.entropy(T, *mu, **kwargs) + sum(mu[i]*self.baryon_minus_antibaryon_comp_density_halfed(i, T, *mu, **kwargs) for i in range(self.num_of_components)) - self.p_eq(T, *mu, **kwargs)

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

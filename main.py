import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve, newton
from scipy.misc import derivative
from scipy.interpolate import InterpolatedUnivariateSpline

from abc import abstractclassmethod

# for storing func results in cache for multiple use
from functools import lru_cache


hbar = 197.3269804


# Main equation of state Class
class Eq_of_state:
    """
    Main functionality of equation of state using its thermodynamic properties.
    To be inherited by child class representing actual equation of state as a function for pressure.
    Partial derivatives calculated here using finite differences method, fast but unstable for second-order derivatives

    Args:
        num_of_eq_in_eos : int
        num_of_components : int
    - describe your equation of state properties

    Child class should define "EoS(root, T, *mu)" method explicitly
    (recommended if your EoS is a system of eq. of single eq. to be solved)
        OR
    define it blank and instead overwrite p_eq(T, *mu) method (recommended if your EoS is a direct expression for p)
    Here
        root : Tuple    - tupe of args used to solve equation of state which is system of equations
                        - for example root = (p, Sigma, K) for ISCT EoS
    """

    def __init__(self, num_of_eq_in_eos, num_of_components):
        self.num_of_eq_in_eos = num_of_eq_in_eos
        self.num_of_components = num_of_components

    @abstractclassmethod
    def EoS(self, root, T, *mu):
        pass

    # solving equation of stated defined via "EoS" method using fsolve or newton method
    @lru_cache(maxsize=512)
    def p_eq(
        self,
        T,
        *mu,
        init=None,
        format="p only",
        method="fsolve",
        xtol=1.49012e-08,
        **kwargs
    ):
        init = init if init else np.full(self.num_of_eq_in_eos, 1.0)
        if method == "fsolve":
            root = fsolve(
                self.EoS,
                init,
                args=(T, *mu),
                maxfev=10000,
                xtol=xtol,
            )
        elif method == "newton":
            root = newton(self.EoS, init, args=(T, *mu), maxiter=10000)
        else:
            raise Exception("Wrong method specified!")

        diff = self.EoS(root, T, *mu)
        if sum(diff[i] ** 2 / root[i] ** 2 for i in range(len(diff))) > 1e-4:
            print("error for T={}, mu={}".format(T, mu[0]))
            return (
                np.nan if format == "p only" else np.full(self.num_of_eq_in_eos, np.nan)
            )
        else:
            return root[0] if format == "p only" else root

    # entropy density (s = ∂p/∂T, μ = constant)
    @lru_cache(maxsize=512)
    def entropy(self, T, *mu, **kwargs):
        return partial_derivative(self.p_eq, 0, [T, *mu], **kwargs)

    # baryon density of a single component  (n_k = ∂p/∂μ_k, T = constant)
    @lru_cache(maxsize=512)
    def density_baryon_comp(self, comp, T, *mu, **kwargs):
        return partial_derivative(self.p_eq, 1 + comp, [T, *mu], **kwargs)

    # total baryon density (n_B = Σ n_k)
    def density_baryon(self, T, *mu, **kwargs):
        return sum(
            self.density_baryon_comp(i, T, *mu, **kwargs) for i in range(len(mu))
        )

    # energy density (ε = T*s + μ*n_B - p)
    def energy(self, T, *mu, **kwargs):
        return (
            T * self.entropy(T, *mu, **kwargs)
            + sum(
                mu[i] * self.density_baryon_comp(i, T, *mu, **kwargs)
                for i in range(len(mu))
            )
            - self.p_eq(T, *mu, **kwargs)
        )

    # entropy by baryon density ratio ( σ  = s / n_B )
    def sigma(self, T, *mu, **kwargs):
        return self.entropy(T, *mu, **kwargs) / self.density_baryon(T, *mu, **kwargs)

    # c_S^2 = (∂p/∂ε)|_σ -> rewritten in terms of partial derivatives with respect to \mu
    # for detailed expalanations see Yakovenko N., MSc thesis 2021, Taras Shevchenko NUK
    def speed_of_s_sq(self, T, *mu, warning=False, **kwargs):
        def d_mu_by_d_T(T, *mu):
            num = partial_derivative(self.sigma, 0, [T, *mu], **kwargs)
            denum = partial_derivative(self.sigma, 1, [T, *mu], **kwargs)
            res = -num / denum

            # dealing with infinite sigma if \mu = 0 hence baryonic density = 0 and constant
            # might be dangerous (may cause untraced errors)
            if np.isnan(res):
                if warning:
                    print("nan value warning for T={}".format(T))
                return 0.0
            else:
                return res

        d_mu_d_T = d_mu_by_d_T(T, *mu)

        num = (
            self.entropy(T, *mu, **kwargs)
            + self.density_baryon(T, *mu, **kwargs) * d_mu_d_T
        )
        denum = (
            partial_derivative(self.energy, 0, [T, *mu], **kwargs)
            + partial_derivative(self.energy, 1, [T, *mu], **kwargs) * d_mu_d_T
        )
        return num / denum

    # speed of sound for multiple chemical potentials
    # NOT TESTED
    def speed_of_s_sq_MULT(self, T, *mu, warning=False, **kwargs):
        def deriv_by_sigma_with_const_T(func, T, *mu):
            num = sum(
                partial_derivative(func, comp + 1, [T, *mu], **kwargs)
                for comp in range(self.num_of_components)
            )
            denum = sum(
                partial_derivative(self.sigma, comp + 1, [T, *mu], **kwargs)
                for comp in range(self.num_of_components)
            )
            res = num / denum

            # dealing with infinite sigma if \mu = 0 hence baryonic density = 0 and constant
            # might be dangerous
            if np.isnan(res):
                if warning:
                    print("nan value warning for T={}".format(T))
                return 0.0
            else:
                return res

        def sigma_deriv_by_T(T, *mu):
            val = partial_derivative(self.sigma, 0, [T, *mu], **kwargs)
            # dealing with infinite sigma if \mu = 0 hence baryonic density = 0 and constant
            # might be dangerous
            if np.isnan(val):
                if warning:
                    print("nan value warning for T={}".format(T))
                return 0.0
            else:
                return val

        sigma_deriv = sigma_deriv_by_T(T, *mu)

        num = (
            partial_derivative(self.p_eq, 0, [T, *mu], **kwargs)
            - deriv_by_sigma_with_const_T(self.p_eq, T, *mu) * sigma_deriv
        )
        denum = (
            partial_derivative(self.energy, 0, [T, *mu], **kwargs)
            - deriv_by_sigma_with_const_T(self.energy, T, *mu) * sigma_deriv
        )
        return num / denum

    # calculating cumulants per volume by definition (κ_j / V = T^(j-1) * (∂^j p)/(∂μ^j) )
    # for following formulas see   arXiv:2103.07365 [nucl-th]  https://arxiv.org/abs/2103.07365v1
    @lru_cache(maxsize=512)
    def cumulant_per_vol(self, j, comp, T, *mu, **kwargs):
        derivative = partial_derivative(self.p_eq, 1 + comp, [T, *mu], n=j, **kwargs)
        return T ** (j - 1) * derivative

    def cumul_lin_ratio(self, T, *mu, **kwargs):
        return (
            T
            * self.cumulant_per_vol(1, 0, T, *mu, **kwargs)
            / mu[0]
            / self.cumulant_per_vol(2, 0, T, *mu, **kwargs)
        )

    def cumul_sq_ratio(self, T, *mu, **kwargs):
        return (
            1.0
            - self.cumulant_per_vol(3, 0, T, *mu, **kwargs)
            * self.cumulant_per_vol(1, 0, T, *mu, **kwargs)
            / self.cumulant_per_vol(2, 0, T, *mu, **kwargs) ** 2
        )


class splined_Eq_of_state:
    """
    Eq_of_state class analogue using spline interpolations.
    To be inherited by child class representing actual equation of state as a function for pressure.
    Partial derivatives calculated here using spline interpolations (default order is 3 but can be increased),
    stable for 2nd and 3rd order derivatives, equal or faster in comparison with original Eq_of_state implementation
    for single variable derivatives but much slower for mixed variable derivatives.

    Args:
        num_of_eq_in_eos : int
        num_of_components : int
    - describe your equation of state properties

    Child class should define "EoS(root, T, *mu)" method explicitly
    (recommended if your EoS is a system of eq. of single eq. to be solved)
        OR
    define it blank and instead overwrite p_eq(T, *mu) method (recommended if your EoS is a direct expression for p)
    Here
        root : Tuple    - tupe of args used to solve equation of state which is system of equations
                        - for example root = (p, Sigma, K) for ISCT EoS
    """

    def __init__(self, num_of_eq_in_eos, num_of_components):
        self.num_of_eq_in_eos = num_of_eq_in_eos
        self.num_of_components = num_of_components

    @abstractclassmethod
    def EoS(self, root, T, *mu):
        pass

    # solves "EoS" equation of state
    @lru_cache(maxsize=512)
    def splined_root(self, T, *mu, init=None, **kwargs):
        init = init if init else np.full(self.num_of_eq_in_eos, 1.0)

        root, infodict, ier, msg = fsolve(
            self.EoS, init, args=(T, *mu), full_output=1, **kwargs
        )
        if ier != 1:
            print(msg)
            return np.full(self.num_of_eq_in_eos, np.nan), infodict
        else:
            return root, infodict

    def splined_p_eq(self, T, *mu, **kwargs):
        root, _ = self.splined_root(T, *mu, **kwargs)
        return root[0]

    def get_spline(
        self, cut, *vars, func=None, splinedata=None, range=5.0, points=5, k=3, **kwargs
    ):
        """
        Returns interpolated spline either
            1) among points in range [var[cut] - range, var[cut] + range]
            2) using splinedata tuple
            CAREFUL to use correct datarange in splinedata which corresponds
                    exactly to the variable you are going to derive

        """
        if not func:
            func = self.splined_p_eq

        if splinedata:
            spline = InterpolatedUnivariateSpline(splinedata[0], splinedata[1], k=k)
        else:
            xdata = np.linspace(vars[cut] - range, vars[cut] + range, points)
            func_data = []
            for x in xdata:
                func_data.append(func(*vars[:cut], x, *vars[cut + 1 :], **kwargs))
            spline = InterpolatedUnivariateSpline(xdata, func_data, k=k)
        return spline

    # entropy density (s = ∂p/∂T, μ = constant)
    def splined_entropy(self, T, *mu, **kwargs):
        """
        Use ONLY temperature T datarange in splinedata
        """
        spline = self.get_spline(0, T, *mu, **kwargs)
        return spline.derivative()(T)

    # baryon density of a single component  (n_k = ∂p/∂μ_k, T = constant)
    def splined_density_baryon_comp(self, comp, T, *mu, **kwargs):
        """
        Use ONLY respective mu[comp] datarange in splinedata
        """
        spline = self.get_spline(1 + comp, T, *mu, **kwargs)
        return spline.derivative()(mu[comp])

    # total baryon density (n_B = Σ n_k)
    def splined_density_baryon(self, T, *mu, splinedata=None, **kwargs):
        """
        Use ONLY mu[0] datarange (mu_b) in splinedata
        splinedata supported here ONLY for single component case
        since multiple dataranges required in multicomponent case (one for each component)
        """
        if self.num_of_components == 1:
            return self.splined_density_baryon_comp(0, T, *mu, **kwargs)
        else:
            if splinedata is not None:
                raise Exception("splinedata not supported here")
            return np.sum(
                self.splined_density_baryon_comp(
                    comp, T, *mu, splinedata=None, **kwargs
                )
                for comp in range(self.num_of_components)
            )

    # energy density (ε = T*s + μ*n_B - p)
    def splined_energy(self, T, *mu, **kwargs):
        """
        Use ONLY temperature T datarange in splinedata
        It will be used for entropy calculations
        """
        return (
            T * self.splined_entropy(T, *mu, **kwargs)
            + np.sum(
                mu[i]
                * self.splined_density_baryon_comp(i, T, *mu, splinedata=None, **kwargs)
                for i in range(self.num_of_components)
            )
            - self.splined_p_eq(T, *mu, **kwargs)
        )

    # entropy by baryon density ratio ( σ  = s / n_B )
    def splined_sigma(self, T, *mu, **kwargs):
        """
        Use ONLY temperature T datarange in splinedata
        It will be used for entropy calculations
        """
        return self.splined_entropy(T, *mu, **kwargs) / self.splined_density_baryon(
            T, *mu, splinedata=None, **kwargs
        )

    # c_S^2 = (∂p/∂ε)|_σ -> rewritten in terms of partial derivatives with respect to \mu
    # for detailed expalanations see Yakovenko N., MSc thesis 2021, Taras Shevchenko NUK
    def splined_speed_of_s_sq(self, T, *mu, **kwargs):
        """
        No splindedata allowed
        """

        def d_mu_by_dt(T, *mu, comp=0):
            """
            here comp=0 specified for now since only single value of mu_B
            (which should be FIRST in mu varible) is supproted for sp of snd
            """
            num = self.get_spline(
                0, T, *mu, func=self.splined_sigma, splinedata=None, **kwargs
            ).derivative()(T)
            denum = self.get_spline(
                1 + comp, T, *mu, func=self.splined_sigma, splinedata=None, **kwargs
            ).derivative()(mu[comp])
            return -num / denum

        d_mu_by_dt = d_mu_by_dt(T, *mu)

        num = (
            self.splined_entropy(T, *mu, **kwargs)
            + self.splined_density_baryon(T, *mu, **kwargs) * d_mu_by_dt
        )

        d_energy_by_dT = self.get_spline(
            0, T, *mu, func=self.splined_energy, splinedata=None, **kwargs
        ).derivative()(T)
        # also only first mu component for now
        d_energy_by_dmu = self.get_spline(
            1, T, *mu, func=self.splined_energy, splinedata=None, **kwargs
        ).derivative()(mu[0])
        denum = d_energy_by_dT + d_energy_by_dmu * d_mu_by_dt

        return num / denum

    # calculating cumulants per volume by definition (κ_j / V = T^(j-1) * (∂^j p)/(∂μ^j) )
    # for following formulas see   arXiv:2103.07365 [nucl-th]  https://arxiv.org/abs/2103.07365v1
    def splined_cumul_per_vol(self, order, comp, T, *mu, **kwargs):
        """
        Use ONLY mu[comp] datarange in splinedata
        """
        spline = self.get_spline(1, T, *mu, **kwargs)
        deriv = spline.derivative(n=order)
        return T ** (order - 1) * deriv(mu[comp])

    def splined_cumul_lin_ratio(self, T, *mu, **kwargs):
        return (
            T
            * self.splined_cumul_per_vol(1, 0, T, *mu, **kwargs)
            / mu[0]
            / self.splined_cumul_per_vol(2, 0, T, *mu, **kwargs)
        )

    def splined_cumul_sq_ratio(self, T, *mu, **kwargs):
        return (
            1.0
            - self.splined_cumul_per_vol(3, 0, T, *mu, **kwargs)
            * self.splined_cumul_per_vol(1, 0, T, *mu, **kwargs)
            / self.splined_cumul_per_vol(2, 0, T, *mu, **kwargs) ** 2
        )


# wrapper for a multivariable function used for partial differentiation
def partial_derivative(func, var=0, point=[], dx=1e-1, order=3, n=1, **kwargs):
    args = point[:]

    def wraps(x):
        args[var] = x
        return func(*args, **kwargs)

    return derivative(wraps, point[var], dx=dx, order=order + (n - 1) * 2, n=n)


# thermal density function
@lru_cache(maxsize=32)
def phi(T, m):
    def func(p):
        return p ** 2 * np.exp(-np.sqrt(p ** 2 + m ** 2) / T)

    integration = quad(func, 0, np.inf)[0]
    return (4 * np.pi) / (2 * np.pi * hbar) ** 3 * integration


def get_changable_dx(var):
    # suitable for T or mu (prob)
    if var < 100.0:
        return 1e-3
    elif var < 250.0:
        return 1e-2
    elif var < 500.0:
        return 1e-1
    else:
        return 1.0

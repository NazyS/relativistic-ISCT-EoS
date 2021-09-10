import numpy as np
from scipy.integrate import quad

from main import Eq_of_state, hbar

# for storing func results in cache for multiple use
from functools import lru_cache


class ISCT(Eq_of_state):
    """
    ISCT equation of state
    Yakovenko et al. Eur. Phys. J. ST 229 (2020) 22-23, 3445-3467
    https://arxiv.org/abs/1910.04889

    Extended with modified function for excluded volume f(p)

    Args:
        each parameter : list of size num_of_components
    """

    def __init__(
        self,
        q1=0.5,
        q2=0.25,
        a1=1.0,
        a2=1.0,
        a3=1.0,
        p_max=150.0,
        delta_p=1.0,
        p0=20.0,
        # parameters for asymptotic f (vdw case) on infinite p   (used in spline interp)
        ad=None,
        qd=0.5,
        ad_factor=1.0,
        # parameters for f spline polynom for connecting f_with_max and f_limit and limits of spline interval
        a_spline=None,
        p_A=np.inf,
        p_B=0.0,
        R=[0.4],
        m=[20.0],
        g=[4.0],
        Alpha=[1.25],
        Beta=[2.45],
        A=[0.5],
        B=[0.5],
    ):
        super().__init__(num_of_eq_in_eos=3, num_of_components=len(m))
        self.m = m
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
        self.R = R if len(R) == self.num_of_components else R * self.num_of_components
        # degeneracy factor
        self.g = g if len(g) == self.num_of_components else g * self.num_of_components
        # asymptotic f func parameters
        self.ad_factor = ad_factor
        self.ad = (
            ad
            if ad
            else self.ad_factor * np.sqrt(self.p0 * self.b(0) / 8.0 / self.m[0])
        )
        self.qd = qd
        # ISCT parameters
        self.Alpha = (
            Alpha
            if len(Alpha) == self.num_of_components
            else Alpha * self.num_of_components
        )
        self.Beta = (
            Beta
            if len(Beta) == self.num_of_components
            else Beta * self.num_of_components
        )
        self.A = A if len(A) == self.num_of_components else A * self.num_of_components
        self.B = B if len(B) == self.num_of_components else B * self.num_of_components

    # excluded volume
    def b(self, component):
        return self.v(component) * 4

    # f(p) function interpolating excluded volume contraction in relativistic case
    def f(self, p, componet):
        return 1 / (
            1
            + sum(
                self.a_parameters[elem] * (p / self.p0) ** self.q_parameters[elem]
                for elem in range(self.num_of_powers)
            )
        )

    def f_with_max(self, p, componet):
        return 1 / (
            1
            + self.a1 * (p / self.p0) ** self.q1
            + (p / self.p0) ** self.q2
            * (self.a2 + self.a3 / ((p - self.p_max) ** 2 + self.delta_p ** 2))
        )

    def f_limit(self, p, componet):
        return 1 / (1 + self.ad * (p / self.p0) ** self.qd)

    def f_spline(self, p, a_spline, componet):
        return np.polynomial.polynomial.polyval(p, a_spline)

    def f_stitched(self, p, a_spline, p_A, p_B, componet):
        if p < p_A:
            return self.f_with_max(p, componet)
        elif p < p_B:
            return self.f_spline(p, a_spline, componet)
        else:
            return self.f_limit(p, componet)

    # Thermal densities
    @lru_cache(maxsize=128)
    def phi(self, T, component):
        return quad(
            lambda p: self.g[component]
            * p ** 2
            / (2 * np.pi ** 2 * hbar ** 3)
            * np.exp(-np.sqrt(p ** 2 + self.m[component] ** 2) / T),
            0,
            np.inf,
        )[0]

    # Values for eigensurface s, eigenvolume v and eigencurvature c for each component
    def s(self, component):
        return 4 * np.pi * self.R[component] ** 2

    def v(self, component):
        return 4 / 3 * np.pi * self.R[component] ** 3

    def c(self, component):
        return self.s(component) / self.R[component]

    # Partial pressure p_k, Ind Surf Tension Sigma_k and Ind Curv Tension K_k for each component
    def pPart(self, p, Sigma, K, T, mu, component):
        return (
            T
            * self.phi(T, component)
            * np.exp(
                (
                    mu / T
                    - self.v(component) * p / T
                    - self.s(component) * Sigma / T
                    - self.c(component) * K / T
                )
                * self.f_stitched(p, self.a_spline, self.p_A, self.p_B, component)
            )
        )

    def SigmaPart(self, p, Sigma, K, T, mu, component):
        return (
            self.A[component]
            * self.R[component]
            * T
            * self.phi(T, component)
            * np.exp(
                (
                    mu / T
                    - self.v(component) * p / T
                    - self.Alpha[component] * self.s(component) * Sigma / T
                    - self.c(component) * K / T
                )
                * self.f_stitched(p, self.a_spline, self.p_A, self.p_B, component)
            )
        )

    def KPart(self, p, Sigma, K, T, mu, component):
        return (
            self.B[component]
            * self.R[component] ** 2
            * T
            * self.phi(T, component)
            * np.exp(
                (
                    mu / T
                    - self.v(component) * p / T
                    - self.Alpha[component] * self.s(component) * Sigma / T
                    - self.Beta[component] * self.c(component) * K / T
                )
                * self.f_stitched(p, self.a_spline, self.p_A, self.p_B, component)
            )
        )

    # Full pressure p,  IST Sigma and ICT K  for system
    def pFull(self, p, Sigma, K, T, *mu):
        return sum(
            self.pPart(p, Sigma, K, T, mu[component], component)
            for component in range(self.num_of_components)
        )

    def SigmaFull(self, p, Sigma, K, T, *mu):
        return sum(
            self.SigmaPart(p, Sigma, K, T, mu[component], component)
            for component in range(self.num_of_components)
        )

    def KFull(self, p, Sigma, K, T, *mu):
        return sum(
            self.KPart(p, Sigma, K, T, mu[component], component)
            for component in range(self.num_of_components)
        )

    def EoS(self, root, T, *mu):
        p, Sigma, K = root
        return (
            self.pFull(p, Sigma, K, T, *mu) - p,
            self.SigmaFull(p, Sigma, K, T, *mu) - Sigma,
            self.KFull(p, Sigma, K, T, *mu) - K,
        )

    # notations for virial expansion
    # Yakovenko et al. Int. J. Mod. Phys. E 29, 11, 2040010 (2020)
    # https://arxiv.org/abs/2004.08693
    def beta_p(self):
        return self.B[0]

    def gamma(self):
        return 3 * (self.Alpha[0] - 1) * (1 - self.beta_p()) * self.v(0)

    def delta(self):
        return 3.0 * self.beta_p() * (self.Beta[0] - 1) * self.v(0)

    def virial_expansion_Sigma(self, rho, order=5):
        """
        expansion of Sigma divided by T

        max order: 5
        """
        if order > 5 or order < 1:
            raise Exception("Wrong order for virial exception specified")

        v = self.v(0)
        gamma = self.gamma()
        delta = self.delta()
        beta_p = self.beta_p()
        coeffs = [
            1,
            4 * v - gamma,
            16 * v ** 2 + 3 * gamma ** 2 / 2 + v * (-14 * gamma - 6 * delta * beta_p),
            64 * v ** 3
            - 8 * gamma ** 3 / 3
            + v ** 2 * (-120 * gamma - 72 * delta * beta_p)
            + v
            * (
                87 * gamma ** 2 / 2
                + (30 * gamma * delta + 27 * delta ** 2 / 2) * beta_p
            ),
            256 * v ** 4
            + 125 * gamma ** 4 / 24
            + v ** 3 * (-832 * gamma - 576 * delta * beta_p)
            + v ** 2
            * (
                624 * gamma ** 2
                + 24 * delta * (26 * gamma + 9 * delta) * beta_p
                + 72 * delta ** 2 * beta_p ** 2
            )
            + v
            * (
                (-111 * gamma ** 2 * delta - 81 * gamma * delta ** 2 - 32 * delta ** 3)
                * beta_p
                - 386 * gamma ** 3 / 3
            ),
        ]

        return (
            self.A[0]
            * self.R[0]
            * sum(rho ** (k + 1) * coeffs[k] for k in range(order))
        )

    # Generating string of EoS parameters
    # generates a string from values of array with corresponding names
    def gen_str_for_array(self, array, name="name", quantity=""):

        # efficient function to check if all values in array are same
        def check_if_equal(array):
            return len(set(array)) == 1

        # if quantity is not empty, then add space in front for good visual separation
        if quantity:
            quantity = " " + quantity

        # if all values in array are same then display its single entry for all components
        if check_if_equal(array):
            string = "${}={:.2f}${}".format(name, array[0], quantity)
        # else display entry for each one separately
        else:
            string = ""
            i = 1
            for elem in array:
                if i != 1:
                    string += ", "
                string += "${}_{:d}=".format(name, i) + "{:.2f}${}".format(
                    elem, quantity
                )
                i += 1
        return string

    def parameters_string(self, format="short"):
        if format == "basic":
            string = (
                self.gen_str_for_array(self.m, name="m", quantity="MeV")
                + ", "
                + self.gen_str_for_array(self.R, name="R", quantity="fm")
            )
            string += ", " + self.gen_str_for_array(self.g, name="g", quantity="")
            return string

        if format == "f parameters":
            string = (
                self.gen_str_for_array(self.q_parameters, name="q", quantity="")
                + ",\n"
                + self.gen_str_for_array(self.a_parameters, name="a", quantity="")
            )
            string += (
                ",\n"
                + self.gen_str_for_array([self.delta_p], name="\\delta p", quantity="")
                + ", "
                + self.gen_str_for_array([self.p_max], name="p_m", quantity="")
                + ", "
                + self.gen_str_for_array([self.p0], name="p_0", quantity="$MeV/fm^{3}$")
            )
            return string

        elif format == "eos parameters":
            string = (
                self.gen_str_for_array(self.Alpha, name="\\alpha", quantity="")
                + ", "
                + self.gen_str_for_array(self.Beta, name="\\beta", quantity="")
                + ",\n"
            )
            string += (
                self.gen_str_for_array(self.A, name="A", quantity="")
                + ", "
                + self.gen_str_for_array(self.B, name="B", quantity="")
            )
            return string
        else:
            print("Wrong format!")

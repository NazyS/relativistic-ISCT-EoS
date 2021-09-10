from main import Eq_of_state


class BagModel(Eq_of_state):
    """
    variation of the MIT Bag Model equation of state

    K. A. Bugaev et al. Physics of Particles and Nuclei Letters, 12, 238–245 (2015)
    https://doi.org/10.1134/S1547477115020065
    """
    def __init__(
        self,
        A0=2.53 * 10 ** (-5),
        A2=1.51 * 10 ** (-6),
        A4=1.001 * 10 ** (-9),
        B=9488.0,
    ):
        super().__init__(num_of_eq_in_eos=1, num_of_components=1)

        self.A0 = A0
        self.A2 = A2
        self.A4 = A4
        self.B = B

    def EoS(self, root, T, *mu):
        pass

    def p_eq(self, T, mu, **kwargs):
        return (
            self.A0 * T ** 4 + self.A2 * T ** 2 * mu ** 2 + self.A4 * mu ** 4 - self.B
        )

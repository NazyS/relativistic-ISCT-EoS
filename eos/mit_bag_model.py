from main import Eq_of_state

class BagModel(Eq_of_state):
    def __init__(
        self,
        A0 = 2.53*10**(-5),
        A2 = 1.51*10**(-6),
        A4 = 1.001*10**(-9),
        B = 9488.,
    ):
        super().__init__()

        self.num_of_eq_in_eos = 1
        self.num_of_components = 1

        self.A0 = A0
        self.A2 = A2
        self.A4 = A4
        self.B = B

    def p_eq(self, T, mu, **kwargs):
        return self.A0*T**4 + self.A2*T**2*mu**2 + self.A4*mu**4 - self.B
import numpy as np
from main import Eq_of_state, phi

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


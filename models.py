import random

from utils import weighted_choice_sub


class HMMBaseClass(object):
    """
    A base class providing common functionality to all HMM object types.
    """
    def __init__(self):
        self._hidden_state = []
        self._symbol_stream = []

    def write_symbols_to_file_as_two_rows(self, filename):
        """
        Write the output symbols to a file.

        This method writes two long rows. The first row is the 'dates' field. The second row is
        the 'ardata' field. All values are separated by semicolons.

        :param filename: The filename to write the data to.
        :type filename: basestring
        """
        f = open(filename, 'w')
        dates_list = [str(i) for i in range(len(self._symbol_stream))]
        ardata_list = [str(i) for i in self._symbol_stream]
        f.write('dates;' + ';'.join(dates_list) + '\n')
        f.write('ardata;' + ';'.join(ardata_list) + '\n')

    def write_symbols_to_file_as_two_columns(self, filename):
        """
        Write the output symbols to a file.

        Each observation symbol is converted to a string and written to a file with the form:

        index, value

        :param filename: The filename to write the data to.
        :type filename: basestring
        """
        f = open(filename, 'w')
        f.write('dates,ardata\n')
        for i, symbol in enumerate(self._symbol_stream):
            f.write('%s,%s\n' % (i, symbol))


class ARHMMGenerator(HMMBaseClass):
    """
    Initialized with parameters of an AR-HMM, then generate data using the generate() method.

    :param N:  The number of hidden states.
    :type  N:  int
    :param A:  The matrix of transition probabilities between hidden states. The value at element
               A[i][j] is the probability of a transition from hidden state i to hidden state j.
    :type  A:  list of lists
    :param C:  The matrix containing p autoregressive coefficients for each of the N hidden states.
               The first rank (the outer list) determines which regime (of N regimes) the model is
               currently in. The second rank contains the coefficients ordered with the with the
               most recent being first (position 0), and the oldest coefficient (the pth value)
               being at position p-1.
    :type  C:  list of lists
    :param R:  The error terms of the autoregressive model.  The first rank (the outer list)
               determines which regime (of N regimes) the model is currently in. The dict contains
               two keys 'mean' and 'std_dev'.
    :type  R:  list of dicts
    :param pi: The initial hidden state probability distribution. A list of length N.
    :type  pi: list
    """

    def __init__(self, N, A, C, R, pi):
        self.N = N
        self.A = A
        self.C = C
        self.R = R
        self.pi = pi
        self.AR_length = [len(regime) for regime in self.C]
        super(ARHMMGenerator, self).__init__()

    def generate_next_data_point(self, coeff, error_mean, error_std_dev):
        error_term = random.normalvariate(error_mean, error_std_dev)
        recent_observations = self._symbol_stream[-abs(len(coeff)):]
        pair = zip(recent_observations, coeff)
        return sum([data * c for data, c in pair]) + error_term

    def generate_data(self, timesteps=1000):
        """
        Generate N timesteps of the AR-HMM.

        This method generates N observations of output symbols and their corresponding hidden
        states according to the AR-HMM parameters.

        :param timesteps: The integer number of timesteps to generate data for. Defaults to 1000.
        :type timesteps: int
        """
        if timesteps < 1:
            raise Exception('get_data() requires the timesteps argument to be >= 1')
        current_state = weighted_choice_sub(self.pi)
        for i in range(timesteps - 1):
            next_point = self.generate_next_data_point(self.C[current_state],
                                                       self.R[current_state]['mean'],
                                                       self.R[current_state]['std_dev'])
            self._hidden_state.append(current_state)
            self._symbol_stream.append(next_point)
            current_state = weighted_choice_sub(self.A[current_state])


class HMMGenerator(object):
    """
    Initialized with the parameters of a HMM, then generate data using the generate() method.

    :param N:  The number of hidden states.
    :type N:   int
    :param M:  The number of distinct symbols per state.
    :type M:   int
    :param A:  The matrix of transition probabilities between hidden states. The value at element
               A[i][j] is the probability of a transition from hidden state i to hidden state j.
    :type A:   list of lists
    :param B:  The matrix containing observation symbol probability distributions for all hidden
               states. The value at element B[j][k] is the probability that the symbol k will be
               observed while in hidden state j.
    :type B:   list of lists
    :param pi: The initial hidden state probability distribution. A list of length N.
    :type pi:  list
    """
    def __init__(self, N, M, A, B, pi):
        self.N = N
        self.M = M
        self.A = A
        self.B = B
        self.pi = pi
        super(ARHMMGenerator, self).__init__()

    def generate_data(self, timesteps=1000):
        """
        Generate N timesteps of the HMM.

        This method generates N observations of output symbols and their corresponding hidden
        states according to the HMM parameters.

        :param timesteps: The integer number of timesteps to generate data for. Defaults to 1000.
        :type timesteps: int
        """
        if timesteps < 1:
            raise Exception('get_data() requires the timesteps argument to be >= 1')
        current_state = weighted_choice_sub(self.pi)
        for i in range(timesteps - 1):
            self._hidden_state.append(current_state)
            self._symbol_stream.append(weighted_choice_sub(self.B[current_state]))
            current_state = weighted_choice_sub(self.A[current_state])

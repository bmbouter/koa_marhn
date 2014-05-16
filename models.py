from utils import weighted_choice_sub


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
        self._hidden_state = []
        self._symbol_stream = []

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
        self._hidden_state.append(current_state)
        self._symbol_stream.append(weighted_choice_sub(self.B[current_state]))
        for i in range(timesteps - 1):
            current_state = weighted_choice_sub(self.A[current_state])
            self._hidden_state.append(current_state)
            self._symbol_stream.append(weighted_choice_sub(self.B[current_state]))

    def write_symbols_to_file(self, filename):
        """
        Write the output symbols to a file.

        Each observation symbol is converted to a string and written to its own line in the file.

        :param filename: The filename to write the data to.
        :type filename: basestring
        """
        f = open(filename, 'w')
        for symbol in self._symbol_stream:
            f.write('%s\n' % symbol)

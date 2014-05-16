from models import HMMGenerator


def viterbi_example():
    """
    Drives the HMMGenerator with an example from wikipedia, and writes output to viterbi_hmm.txt

    The example is taken from:    http://en.wikipedia.org/wiki/Viterbi_algorithm#Example

    states = ('Healthy', 'Fever')

    observations = ('normal', 'cold', 'dizzy')

    start_probability = {'Healthy': 0.6, 'Fever': 0.4}
    transition_probability = {
        'Healthy' : {'Healthy': 0.7, 'Fever': 0.3},
        'Fever' : {'Healthy': 0.4, 'Fever': 0.6}
    }

    emission_probability = {
        'Healthy' : {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
        'Fever' : {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6}
    }
    """

    # Mapping the strings to integers
    # healthy = 0; fever = 1
    # normal = 0; cold = 1; dizzy = 1;
    N = 2
    M = 3
    A = [None for i in xrange(N)]
    A[0] = [0.7, 0.3]
    A[1] = [0.4, 0.6]
    B = [None for i in xrange(N)]
    B[0] = [0.5, 0.4, 0.1]
    B[1] = [0.1, 0.3, 0.6]
    pi = [0.8, 0.2]
    my_hmm = HMMGenerator(N, M, A, B, pi)
    my_hmm.generate_data(100)
    my_hmm.write_symbols_to_file('viterbi_hmm.txt')

if __name__ == "__main__":
    viterbi_example()

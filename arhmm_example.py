from models import ARHMMGenerator


def arhmm_example():
    N = 2
    A = [None for i in xrange(N)]
    A[0] = [0.7, 0.3]
    A[1] = [0.4, 0.6]
    C = [None for i in xrange(N)]
    C[0] = [-0.5, 0.7, 0.1]
    C[1] = [0.1, -0.1, 0.2]
    R = [{'mean': 1, 'std_dev': 2}, {'mean': 3, 'std_dev': 3}]
    pi = [0.8, 0.2]
    my_ar_hmm = ARHMMGenerator(N, A, C, R, pi)
    iterations = 280
    my_ar_hmm.generate_data(iterations)
    my_ar_hmm.write_symbols_to_file('arhmm_example_%s.csv' % iterations)

if __name__ == "__main__":
    arhmm_example()

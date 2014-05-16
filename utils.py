import random


def weighted_choice_sub(weights):
    """
    Implements a weighted random selection in Python.

    Given a list of weights, it returns an index randomly, according to these weights.

    Originally from:  http://eli.thegreenplace.net/2010/01/22/weighted-random-generation-in-python/

    :param weights: A list of probabilities. sum(weights) should equal 1.0
    :type weights:  list
    :return: The integer index of the selection.
    :rtype: int
    """
    rnd = random.random() * sum(weights)
    for i, w in enumerate(weights):
        rnd -= w
        if rnd < 0:
            return i

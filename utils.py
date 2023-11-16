import numpy as np


def mse(predicted, actual):
    errors = []
    for i in range(len(predicted)):
        errors.append((predicted[i] - actual[i])**2)
    return sum(errors)/len(errors)


def score(predicted, actual):
    u = sum([(actual[i] - predicted[i])**2 for i in range(len(predicted))])
    actual_mean = np.mean(actual)
    v = sum([(actual[i] - actual_mean)**2 for i in range(len(predicted))])
    return 1 - u/v

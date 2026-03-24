import math


def harmonic_number(n):
    if n <= 0:
        return 0.0
    
    return math.log(n) + 0.5772156649


def c_factor(n):
    if n <= 1:
        return 0.0
    
    return 2.0 * harmonic_number(n - 1) - (2.0 * (n - 1) / n)


def anomaly_score(expected_path_length, sample_size):
    c_n = c_factor(sample_size)
    if c_n <= 0:
        return 0.0
    
    return 2.0 ** (-(expected_path_length / c_n))
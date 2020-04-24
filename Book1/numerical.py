import numpy as np
import matplotlib.pylab as plt


def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)


def function_1(x):
    return 0.01*x**2 + 0.1*x


def function_2(x):
    return np.sum(x**2)


def function_tmp1(x0):
    return x0*x0 + 4.0**2.0


def function_tmp2(x1):
    return 3.0**2.0 + x1*x1


print(numerical_diff(function_1, 5))
print(numerical_diff(function_1, 10))
print(numerical_diff(function_tmp1, 3.0))
print(numerical_diff(function_tmp2, 4.0))

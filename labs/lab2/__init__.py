import numpy as np
from numpy import log as ln
import sympy as sp
from scipy.integrate import quad

x = sp.symbols('x')
table = []

def exact_integral(f, a, b):
    return quad(f, a, b)[0]

def simpsons_rule(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    arr = [[i, x[i], y[i]] for i in range(n+1)]
    return h/3 * (y[0] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-1:2]) + y[-1]), arr


def create_function(expression, for_der=False):
    parsed_expression = sp.sympify(expression)
    if for_der:
        return parsed_expression
    f = sp.lambdify(x, parsed_expression, modules=[{"log": np.log, "sqrt": np.sqrt}])
    return f


def derivative(f_str, a, h):
    f = create_function(f_str)
    f_d = create_function(f_str, for_der=True)
    d = f_d.diff(x).subs(x, a)
    res = (f(a + h) - f(a - h))/(2*h)
    return res, abs(res - d)

def integral(f_str, a, b, n):
    f = create_function(f_str)
    exact_result = exact_integral(f, a, b)
    result, table_integral = simpsons_rule(f, a, b, n)
    return result, table_integral, abs(result-exact_result)

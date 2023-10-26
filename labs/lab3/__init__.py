import numpy as np
from numpy import exp
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


def analytic_sol(x):
    return -1*exp(2*x) + 0.5 * exp(3*x) + exp(x)/2
# Функція, яка визначає систему рівнянь
def f(x, y, z):
    dydx = z
    dzdx = 5 * y - 6 * z + np.exp(x)
    return dydx, dzdx

# Метод Рунге-Кутта четвертого порядку
def runge_kutta4(h, x0, y0, z0, x_end):
    x_values = [x0]
    y_values = [y0]
    z_values = [z0]

    while x0 < x_end:
        dy1, dz1 = f(x0, y0, z0)
        dy2, dz2 = f(x0 + h / 2, y0 + h / 2 * dy1, z0 + h / 2 * dz1)
        dy3, dz3 = f(x0 + h / 2, y0 + h / 2 * dy2, z0 + h / 2 * dz2)
        dy4, dz4 = f(x0 + h, y0 + h * dy3, z0 + h * dz3)

        x0 += h
        y0 += (h / 6) * (dy1 + 2 * dy2 + 2 * dy3 + dy4)
        z0 += (h / 6) * (dz1 + 2 * dz2 + 2 * dz3 + dz4)

        x_values.append(x0)
        y_values.append(y0)
        z_values.append(z0)

    return x_values, y_values

# Початкові умови
x0 = 0
y0 = 0
z0 = 0

# Крок інтегрування та кінцева точка
h = 0.02
x_end = 0.2

x_values, y_values = runge_kutta4(h, x0, y0, z0, x_end)
y_fact = [analytic_sol(x) for x in x_values]


# Виведення результатів
def get_results_l3():
    results = []
    for i in range(len(x_values)):
        results.append(f'x = {x_values[i]:.2f}, y = {y_values[i]:.6f}, delta = {abs(y_fact[i]-y_values[i]):6f}')
    # Графік
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, label="Чисельний розв'язок", color='purple', linewidth=2)
    plt.plot(x_values, y_fact, label="Аналітичний розв'язок", color='blue', linestyle='--', linewidth=2)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Чисельний та аналітичний розв'язок")
    plt.legend()
    plt.grid(True)
    plt.show()
    return results



# Display the plot

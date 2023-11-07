import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

x = np.array([1, 2])  # Початкова точка
temp_x = x.copy()  # Тимчасова змінна для збереження поточної точки
h = 0.01  # Крок для обчислення похідних
epsilon = 1e-6  # Точність
l0 = 0  # Початкове значення параметру для методу золотого перетину
param = 0.01  # Параметр для визначення кроку

def f(x1, x2):
    return x1**2 - x1*x2 + x2**2  # Функція, яку ми мінімізуємо

def get_norm(var):
    return np.sqrt(var[0] ** 2 + var[1] ** 2)  # Функція для обчислення норми вектора

def get_grad_center(t_x, f=f, h=h):
    # Обчислення градієнта функції в точці
    dfdx1 = (f(t_x[0] + h, t_x[1]) - f(t_x[0] - h, t_x[1])) / (2 * h)
    dfdx2 = (f(t_x[0], t_x[1] + h) - f(t_x[0], t_x[1] - h)) / (2 * h)
    return np.array([dfdx1, dfdx2])

def sven_intervals(delta, f, l0, s):
    # Пошук інтервалу, де знаходиться мінімум
    global temp_x
    ls = [l0]

    def fn(lam):
        return f(temp_x[0] + lam * s[0], temp_x[1] + lam * s[1])

    def new_lambda(old, k, forward):
        new_l = old + delta * 2 ** k if forward else old - delta * 2 ** k
        return new_l

    l1_front = l0 + delta
    l1_back = l0 - delta
    f_l0 = fn(l0)
    f_l1_front = fn(l1_front)
    f_l1_back = fn(l1_back)
    if f_l1_back > f_l0 and f_l1_front > f_l0:
        return -delta, delta
    forward = f_l1_front < f_l1_back
    ls.append(l1_front if forward else l1_back)
    while fn(ls[-1]) < fn(ls[-2]):
        ls.append(new_lambda(ls[-1], len(ls) - 1, forward))
    ls.append((ls[-1] + ls[-2]) / 2)
    ls.sort()
    y = [fn(i) for i in ls]
    result_ind = y.index(min(y))
    return ls[result_ind - 1], ls[result_ind + 1]

def golden_section_search(f, a, b, eps, s):
    # Метод золотого перетину для пошуку точки мінімуму
    global temp_x
    def fn(lam):
        return f(temp_x[0] + lam * s[0], temp_x[1] + lam * s[1])

    golden_ratio = (np.sqrt(5) - 1) / 2
    x1 = b - golden_ratio * (b - a)
    x2 = a + golden_ratio * (b - a)
    while abs(b - a) > eps:
        if fn(x1) < fn(x2):
            b = x2
            x2 = x1
            x1 = b - golden_ratio * (b - a)
        else:
            a = x1
            x1 = x2
            x2 = a + golden_ratio * (b - a)
    return (b + a) / 2

def one_iteration(f, l0, eps):
    # Один крок методу найшвидшого спуску
    global temp_x
    grad = get_grad_center(temp_x)
    s = - grad
    delta = param * np.linalg.norm(temp_x.astype(float)) / np.linalg.norm(s.astype(float))
    a, b = sven_intervals(delta, f, l0, s)
    lamb = golden_section_search(f, a, b, eps, s)
    new_x = temp_x + lamb * s
    return new_x, lamb

def draw_trajectory(df, title):
    # Візуалізація траєкторії пошуку мінімуму
    x1 = df['x1'].to_numpy()
    x2 = df['x2'].to_numpy()
    x = np.linspace(-2.5, 2.5, 100)
    y = x
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    fig, ax = plt.subplots()
    ax.contour(X, Y, Z, levels=50, cmap='coolwarm')
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-2.5, 2.5])

    plt.plot(x1, x2, 'o-', linewidth=2, c="purple", markerfacecolor="orange", markersize=10, label="general")
    plt.title(title)
    plt.grid()
    plt.show()

def mns():
    # Головна функція методу найшвидшого спуску
    results_df = pd.DataFrame(columns=["i", "x1", "x2", "f(x1,x2)", "grad_norm"])
    global temp_x
    i = 0
    my_eps= False
    while not my_eps and i<10000:
        grad = get_grad_center(temp_x)
        results_df.loc[i] = [i, temp_x[0], temp_x[1], f(temp_x[0], temp_x[1]), np.linalg.norm(grad)]
        new_x, step = one_iteration(f, l0, epsilon)
        my_eps = np.linalg.norm(grad) <= epsilon
        temp_x = new_x
        i += 1
    draw_trajectory(results_df, "Траекторія пошуку мінімуму")
    return results_df.values


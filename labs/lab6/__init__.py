import numpy as np
import pulp
from scipy.optimize import linprog

C = np.array([
    [10, 17, 9, 20, 30],
    [15, 4, 24, 26, 26],
    [22, 24, 30, 27, 29],
    [25, 12, 11, 24, 23]
])
a = np.array([15, 15, 19, 11])
b = np.array([9, 24, 9, 9, 9])
C_l = np.array([
    [7, 19, 7, 12, 18],
    [17, 11, 7, 13, 11],
    [1, 13, 19, 18, 12],
    [8, 14, 11, 3, 11]
])

D_l = np.array([
    [20, 6, 15, 22, 25],
    [2, 5, 2, 3, 4],
    [20, 1, 3, 15, 8],
    [40, 5, 6, 2, 10]
])
a_l = np.array([80, 12, 38, 45])
b_l = np.array([75, 10, 20, 40, 30])


def transp_default(a=a, b=b, C=C):
    m, n = len(a), len(b)
    C_vec = C.reshape((m*n, 1), order='F')
    A1 = np.kron(np.ones((1, n)), np.identity(m))
    A2 = np.kron(np.identity(n), np.ones((1, m)))
    A = np.vstack([A1, A2])
    b = np.hstack([a, b])
    res = linprog(C_vec, A_eq=A, b_eq=b)
    return res.x.reshape((m,n), order='F'), res.fun



def transp_with_limits(C=C_l, D=D_l, a=a_l, b=b_l):
    lp = pulp.LpProblem("Limited_Transportation_Problem", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("X", [(i, j) for i in range(len(a)) for j in range(len(b))],
                              lowBound=0, cat=pulp.LpInteger)
    lp += pulp.lpSum(C[i][j] * x[(i, j)] for i in range(len(a)) for j in range(len(b)))
    for i in range(len(a)):
        lp += pulp.lpSum(x[(i, j)] for j in range(len(b))) <= a[i]
    for j in range(len(b)):
        lp += pulp.lpSum(x[(i, j)] for i in range(len(a))) == b[j]
    for i in range(len(a)):
        for j in range(len(b)):
            lp += x[(i, j)] <= D[i][j]
    lp.solve()
    results = np.array([[x[(i, j)].varValue for j in range(len(b))] for i in range(len(a))])
    return results, pulp.value(lp.objective)


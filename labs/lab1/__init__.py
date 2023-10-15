import numpy as np

<<<<<<< HEAD
# Перевірка чи містить головна діагональ нульові елементи
=======
>>>>>>> 0d231ae06451685b6c4d21fca306279ea46384fd
def if_null_on_diagonal(matrix):
    n = len(matrix)
    for i in range(n):
        if matrix[i][i] == 0.0:
            return i+1
    return False

<<<<<<< HEAD
# Зміна рядків/стовпців місцями у випадку, коли нулі стоять на головній діагоналі
=======

>>>>>>> 0d231ae06451685b6c4d21fca306279ea46384fd
def relocate_matrix_rows(matrix, j):
    def null_case(indxs):
        if len(indxs) == 1:
            ind = indxs[0]
            matrix[[ind, j]] = matrix[[j, ind]]
            return 1
        return 0

    rows, cols = matrix.shape
    column, row = matrix[:, j], matrix[j]
    nonzero_indexes, nonzero_indexes_row = np.nonzero(column)[0], np.nonzero(row)[0]
    resp_cols, resp_rows = null_case(nonzero_indexes), null_case(nonzero_indexes_row)
    if resp_cols or resp_rows:
        return 0

    for i in range(rows):
        if i != j and matrix[i, j] != 0 and matrix[j, i] != 0:
            matrix[[i, j]] = matrix[[j, i]]
            return 1
    return 100

<<<<<<< HEAD
# Головна функція для перевірки нульових елементів на діагоналі
=======

>>>>>>> 0d231ae06451685b6c4d21fca306279ea46384fd
def check_diag_nulls(a):
    i_to_relocate = if_null_on_diagonal(a)
    counter = 0
    while i_to_relocate > 0:
        i_to_relocate -= 1
        counter += relocate_matrix_rows(a, i_to_relocate)
        i_to_relocate = if_null_on_diagonal(a)
<<<<<<< HEAD
        # Відпрацьовує у випадку, коли є повністю нульовий рядок або коли неможливо позбутись нуля на головній діагоналі
=======
>>>>>>> 0d231ae06451685b6c4d21fca306279ea46384fd
        if counter > 100:
            raise Exception('Impossible to find the solution!')


def gaussian_elimination(matrix):
    a = matrix.copy()
    n = len(a)
    check_diag_nulls(a)
<<<<<<< HEAD
    # Прямий хід
=======
>>>>>>> 0d231ae06451685b6c4d21fca306279ea46384fd
    for i in range(n):
        for j in range(i + 1, n):
            ratio = a[j][i] / a[i][i]

            for k in range(n + 1):
                a[j][k] = a[j][k] - ratio * a[i][k]
    n = len(a)
    solutions = np.zeros(n)
<<<<<<< HEAD
    # Зворотній хід
=======
>>>>>>> 0d231ae06451685b6c4d21fca306279ea46384fd
    solutions[n - 1] = a[n - 1][n] / a[n - 1][n - 1]
    for i in range(n - 2, -1, -1):
        solutions[i] = a[i][n]

        for j in range(i + 1, n):
            solutions[i] = solutions[i] - a[i][j] * solutions[j]

        solutions[i] = solutions[i] / a[i][i]
    return solutions

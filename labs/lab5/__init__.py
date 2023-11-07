import pulp


def get_solution(A, c, b):

    # Створення задачі лінійного програмування
    lp_problem = pulp.LpProblem("Максимізація_Прибутку", pulp.LpMaximize)

    # Визначення рішень
    x1 = pulp.LpVariable("x1", lowBound=0, cat="Integer")
    x2 = pulp.LpVariable("x2", lowBound=0, cat="Integer")
    x3 = pulp.LpVariable("x3", lowBound=0, cat="Integer")

    # Цільова функція
    lp_problem += pulp.lpDot(c, [x1, x2, x3]), "Цільова_функція"

    # Обмеження
    for i in range(len(b)):
        lp_problem += pulp.lpDot(A[i], [x1, x2, x3]) <= b[i], f"Обмеження{i+1}"

    # Розв'язання задачі лінійного програмування
    lp_problem.solve()

    # Виведення математичної моделі
    res_model = []
    res_model.append("Математична модель задачі лінійного програмування:")
    res_model.append("Максимізувати:")
    res_model.append(f"Z ={pulp.lpDot(c, [x1, x2, x3])}")
    res_model.append("Обмеження:")
    for i in range(len(b)):
        res_model.append(f"{pulp.lpDot(A[i], [x1, x2, x3])} <= {b[i]}")

    # Виведення результатів
    if pulp.LpStatus[lp_problem.status] == "Optimal":
        res_model.append(f"Максимальний прибуток: {pulp.value(lp_problem.objective)}")
        res_model.append(f"Кількість x1: {int(pulp.value(x1))}")
        res_model.append(f"Кількість x2: {int(pulp.value(x2))}")
        res_model.append(f"Кількість x3: {int(pulp.value(x3))}")
    else:
        res_model.append("Рішення не знайдено.")
    return res_model

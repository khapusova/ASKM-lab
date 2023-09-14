from flask import *
import numpy as np
from labs.lab1 import gaussian_elimination
app = Flask(__name__)

selected_value = 4
# matrix = np.array([['', '']])
matrix = np.array([[3, 1, -1, 2, 6],
             [-5, 1, 3, -4, -12],
             [2, 0, 1, -1, 1],
             [1, -5, 3, -3, 3]])

errors, sums, answers = [], [], []

@app.route('/lab1', methods=['GET', 'POST'])
def lab1():
    global selected_value, matrix, errors, answers, sums
    if request.method == 'POST':
        if request.form["btn"] == "Submit":
            selected_value = int(request.form.get('select_option'))
            matrix = np.array([np.array(['' for j in range(selected_value + 1)]) for i in range(selected_value)])
            errors, answers, sums = [], [], []
        else:
            try:
                if len(matrix) == 0 and selected_value == 1:
                    raise ValueError
                matrix = np.array([np.array([request.form.get(f'input_{i}_{j}') for j in range(selected_value + 1)])
                                for i in range(selected_value)])
                matrix = np.array(matrix, dtype=float)
                errors = []
                answers = gaussian_elimination(matrix)

                matrix_without_last_column = matrix[:, :-1]
                a_multiplied = matrix_without_last_column * answers
                sums = np.sum(a_multiplied, axis=1)
            except ValueError:
                errors = ['You entered wrond data!']
            except ZeroDivisionError:
                errors = ['It`s impossible to divide number by zero!']
            except Exception as e:
                errors = [e]
        return render_template('index.html', values=np.arange(1, 11), n=selected_value,
                               matrix=matrix, len_matr=len(matrix), errors=errors, answers=answers,
                               len_ans=len(answers), sums=sums, len_sums=len(sums))
    return render_template('index.html', values=np.arange(1, 11), n=selected_value,
                           matrix=matrix, len_matr=len(matrix), errors=errors, answers=answers,
                            len_ans=len(answers), sums=sums, len_sums=len(sums))


if __name__ == '__main__':
    app.run()

from flask import *
import numpy as np
from labs.lab1 import gaussian_elimination
<<<<<<< HEAD
from labs.lab2 import integral, derivative

=======
>>>>>>> 0d231ae06451685b6c4d21fca306279ea46384fd
app = Flask(__name__)

selected_value = 4
# matrix = np.array([['', '']])
matrix = np.array([[3, 1, -1, 2, 6],
             [-5, 1, 3, -4, -12],
             [2, 0, 1, -1, 1],
             [1, -5, 3, -3, 3]])

errors, sums, answers = [], [], []

<<<<<<< HEAD
#lab 2

func_text, func_text2 = "1/(x*(x**2 + 0.25)**(1/2))", "-2 * log((0.5 + (x**2 + 0.25)**(1/2))/x)"
from_inp, to_inp, n, x, h = 1, 2, 50, 1, 0.02
answer_int, answer_der, delta_int, delta_der, table_integral = False, False, False, False, []

=======
>>>>>>> 0d231ae06451685b6c4d21fca306279ea46384fd
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


<<<<<<< HEAD



@app.route('/lab2', methods=['GET', 'POST'])
def lab2():
    global func_text, func_text2, answer_int, from_inp, to_inp, n,\
        table_integral, x, h, answer_der, delta_int, delta_der, errors
    if request.method == 'POST':
        try:
            new_text = request.form.get('function_inp')
            new_text2 = request.form.get('function_inp2')
            from_inp = float(request.form.get('from_inp'))
            to_inp = float(request.form.get('to_inp'))
            n = int(request.form.get('n'))
            x = float(request.form.get('x'))
            h = float(request.form.get('h'))

            if len(new_text) != 0:
                func_text = new_text
                answer_int, table_integral, delta_int = integral(func_text, from_inp, to_inp, n)
                # answer_der = derivative(func_text, x, h = (to_inp - from_inp) / n)
            if len(new_text2) != 0:
                func_text2 = new_text2
                answer_der, delta_der = derivative(func_text2, x, h)
            return render_template('lab2.html', func_text=func_text, func_text2=func_text2, answer_int=answer_int,
                                   from_inp=from_inp, to_inp=to_inp, n=n, table_integral=table_integral,
                                   answer_der=answer_der, x=x, h=h, delta_der=delta_der, delta_int= delta_int, errors=errors)
        except Exception as e:
            errors = [e]
        return render_template('lab2.html', func_text=func_text,  func_text2=func_text2, answer_int=answer_int,
                           from_inp=from_inp, to_inp=to_inp, n=n, table_integral=table_integral,
                           answer_der=answer_der, x=x, h=h, delta_der=delta_der, delta_int = delta_int, errors=errors)


=======
>>>>>>> 0d231ae06451685b6c4d21fca306279ea46384fd
if __name__ == '__main__':
    app.run()

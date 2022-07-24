import os
from subprocess import Popen, PIPE
import pandas as pd
import numpy as np
import holoviews as hv
import panel as pn
import hvplot.pandas  # noqa
hv.extension('bokeh', 'matplotlib')

WIDTH = 800
DEFAULT_EXE_PATH = r'C:\Users\Sergei\Documents\Jupyter\Release\EApp.exe'

M_SYS_VALUES = (np.geomspace(1.1, 11, 20) - 1.0).round(1)  # or np.arange(0.1, 8, step=0.2)
A_PARAMS = np.array([39.098, 1.0, 1.0, 3.3417, 200.0, 78.4])
B_PARAMS = np.array([35.453, -1.0, 1.0, 2.7560, 170.0, 78.4])
K_MATRIX = np.array([[0., 0., 0.],
                     [0.2, 1., 0.],
                     [-0.25, 0.064, 1.]])  # Maybe transposed


def transform_params_to_string(m_sys, a_params, b_params, k_matrix):
    a_str = ' '.join(map(str, a_params))
    b_str = ' '.join(map(str, b_params))
    k_str = ' '.join(map(str, k_matrix.flatten()))
    return f'{m_sys} {a_str} {b_str} {k_str}'


def fortran_wrapper(m_sys, a_params, b_params, k_matrix, exe_path):
    p = Popen([exe_path], stdout=PIPE, stdin=PIPE, stderr=PIPE)
    arguments_as_str = transform_params_to_string(m_sys, a_params, b_params, k_matrix)
    stdout_data = p.communicate(input=arguments_as_str.encode())[0].decode()
    output = np.array([float(i) for i in stdout_data.split()][1:])  # first output obsolete
    p.terminate()
    return output


exe_selector = pn.widgets.FileSelector(directory='~', only_files=True, file_pattern='*.exe',
                                       value=[DEFAULT_EXE_PATH], width=WIDTH)
m_sys_widget = pn.widgets.ArrayInput(name='m_sys values', value=M_SYS_VALUES)
a_widget = pn.widgets.ArrayInput(name='A params', value=A_PARAMS)
b_widget = pn.widgets.ArrayInput(name='B params', value=B_PARAMS)
k_widget = pn.widgets.ArrayInput(name='K matrix', value=K_MATRIX)


@pn.depends(m_sys_widget, a_widget, b_widget, k_widget, exe_selector)
def interactive_plot(m_sys_values, a_params, b_params, k_matrix, files):
    if len(files) != 1:
        return pn.pane.Markdown('## Please select exactly one .exe file', width=WIDTH)
    exe_path = files[0]
    out_arrays = [fortran_wrapper(m_sys, a_params, b_params, k_matrix, exe_path) for m_sys in m_sys_values]
    df = pd.DataFrame(out_arrays, columns=['out_1', 'out_2', 'out_3', 'out_4'])
    df.index = m_sys_values
    df.index.name = 'm_sys'
    plots = df.hvplot(grid=True) * df.hvplot.scatter()
    return plots


def create_app():
    material = pn.template.MaterialTemplate(title='Some chemistry')
    inputs = pn.Column(m_sys_widget, a_widget, b_widget, k_widget, width=WIDTH)
    main = pn.Column(pn.pane.Markdown('## Inputs'),
                     inputs,
                     pn.pane.Markdown('## Plots'),
                     interactive_plot,
                     )
    material.main.append(main)
    return material


if __name__ == '__main__':
    assert os.path.isfile(DEFAULT_EXE_PATH), f'Cannot find {DEFAULT_EXE_PATH}, please set variable DEFAULT_EXE_PATH'
    app = create_app()
    pn.serve(app)

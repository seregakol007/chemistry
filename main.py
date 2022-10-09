import os
from subprocess import Popen, PIPE
import pandas as pd
import numpy as np
import holoviews as hv
import panel as pn
import hvplot.pandas  # noqa
from bokeh.models.widgets.tables import NumberFormatter
hv.extension('bokeh', 'matplotlib')

_formatter = NumberFormatter(format='0.000')
_formatter_short = NumberFormatter(format='0.0')

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


def fortran_wrapper(m_sys, a_params, b_params, k_matrix):
    p = Popen([DEFAULT_EXE_PATH], stdout=PIPE, stdin=PIPE, stderr=PIPE)
    arguments_as_str = transform_params_to_string(m_sys, a_params, b_params, k_matrix)
    stdout_data = p.communicate(input=arguments_as_str.encode())[0].decode()
    output = np.array([float(i) for i in stdout_data.split()][1:])  # first output obsolete
    p.terminate()
    return output


m_sys_df = pd.DataFrame({'m_sys': M_SYS_VALUES}).transpose()
m_sys_df.index.name = 'N'
m_sys_widget = pn.widgets.DataFrame(m_sys_df, formatters={k: _formatter_short for k in m_sys_df.columns}, auto_edit=True,
                                reorderable=False, sortable=False, width=WIDTH)

df_components = pd.DataFrame({'A': A_PARAMS, 'B': B_PARAMS}).transpose()
df_components.index.name = 'component'
df_components.columns = [f'feature_{i}' for i in df_components.columns]

components_widget = pn.widgets.DataFrame(df_components, formatters={k: _formatter for k in df_components.columns}, auto_edit=True,
                                reorderable=False, sortable=False, width=WIDTH)

df_k = pd.DataFrame(K_MATRIX)
df_k.index.name = 'K_MATRIX'

k_widget = pn.widgets.DataFrame(df_k, formatters={k: _formatter for k in df_k.columns}, auto_edit=True,
                                reorderable=False, sortable=False, width=WIDTH)


@pn.depends(m_sys_widget, components_widget, k_widget)
def interactive_plot(m_sys_values, components_matrix, k_matrix):
    m_sys_values = m_sys_values.values[0]
    a_params = components_matrix.iloc[0]
    b_params = components_matrix.iloc[1]
    k_matrix = k_matrix.to_numpy()
    out_arrays = [fortran_wrapper(m_sys, a_params, b_params, k_matrix) for m_sys in m_sys_values]
    df = pd.DataFrame(out_arrays, columns=['out_1', 'out_2', 'out_3', 'out_4'])
    df.index = m_sys_values
    df.index.name = 'm_sys'
    plots = [(df[i].hvplot(grid=True) * df[i].hvplot.scatter()).opts(width=WIDTH // 2, show_legend=False) for i in df.columns]
    return pn.Column(pn.Row(plots[0], plots[1]), pn.Row(plots[2], plots[3]), df)


def create_app():
    material = pn.template.MaterialTemplate(title='Some chemistry')
    inputs = pn.Column(m_sys_widget, components_widget, k_widget, width=WIDTH)
    main = pn.Column(pn.pane.Markdown('## Inputs'),
                     inputs,
                     pn.pane.Markdown('## Plots'),
                     interactive_plot,
                     )
    material.main.append(main)
    return material


assert os.path.isfile(DEFAULT_EXE_PATH), f'Cannot find {DEFAULT_EXE_PATH}, please set variable DEFAULT_EXE_PATH'


if __name__ == '__main__':
    assert os.path.isfile(DEFAULT_EXE_PATH), f'Cannot find {DEFAULT_EXE_PATH}, please set variable DEFAULT_EXE_PATH'
    app = create_app()
    pn.serve(app)

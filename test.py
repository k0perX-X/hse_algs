import numpy as np
import pandas as pd
from typing import List

def simplex_max(table: np.ndarray, columns: List[str] = None, rows: List[str] = None) -> dict:
    iter = 0
    columns = ['free'] + [f'X{i}' for i in range(1, table.shape[1])] if columns is None else columns
    rows = [f'X{i + table.shape[1]}' for i in range(table.shape[0] - 1)] + ['F'] if rows is None else rows
    table = np.array(table).copy().astype(float)
    leading_col = None

    start_table = pd.DataFrame(table, columns=columns, index=rows)

    print(iter)
    print(start_table)

    while np.sum(table[-1] >= 0) < table.shape[1]:
        iter += 1
        print(iter)

        table2 = table.copy()
        if leading_col is not None:
            table2[0, leading_col] = 0

        leading_col = abs(table2[:, 1:]).argmax() % (table.shape[1] - 1) + 1 \
            if sum(table2[-1] <= 0) > 1 or table2[-1, 0] < 0 else np.where(table2[-1] <= 0)[0][0]

        col = np.divide(table2.T[0], table2.T[leading_col])
        col[np.logical_and(table2.T[0] >= 0, table2.T[leading_col] < 0)] = np.inf
        # if sum(np.isinf(col)) == table.shape[0]:
        #     print(table)
        #     print(table2)
        #     print(leading_col)
        #     print(col)
        #     raise Exception
        leading_row = abs(col).argmin()

        # print(table)
        # print(col)
        # print(leading_col, leading_row)

        table2[0] = table[leading_row] / table[leading_row, leading_col]
        table2[0, leading_col] = 1 / table[leading_row, leading_col]
        auxiliary = -table.T[leading_col]
        auxiliary = np.delete(auxiliary, leading_row)
        table = np.delete(table, leading_row, axis=0)
        table.T[leading_col] = 0

        for i in range(auxiliary.shape[0]):
            table2[i + 1] = table2[0] * auxiliary[i] + table[i]

        table = table2

        changed_col = columns[leading_col]
        columns[leading_col] = rows.pop(leading_row)
        rows.insert(0, changed_col)

        print(pd.DataFrame(table, columns=columns, index=rows))
        print(auxiliary)
        print(leading_col)
        # input()

    def get_value(value: str, table: pd.DataFrame, values: dict) -> float:
        if value in table.index:
            row = table.loc[value]
            out = -row.iloc[0]
            for i in range(1, len(row.index)):
                out += row.iloc[i] * values[row.index[i]]
            return float(out)
        else:
            row = table.T.loc[value]
            out = 0
            for i in range(0, len(row.index) - 1):
                out += row.iloc[i] * values[row.index[i]]
            return float(-out)

    values = pd.DataFrame(table, columns=columns, index=rows).iloc[:, 0].to_dict()
    for i in range(1, len(columns)):
        print(values)
        values[columns[i]] = get_value(columns[i], start_table, values)
    return values


U = {
    (4, 'Маг'): {
        'Z': 5,
        'D': 3,
        'K': 4,
    },
    (5, 'Рыцарь'): {
        'Z': 4,
        'D': 5,
        'K': 7,
    },
    (50, 'Дракон'): {
        'Z': 50,
        'D': 35,
        'K': 40,
    }
}

R = {
    'Z': 400,
    'D': 300,
    'K': 500,
}
Xs = [[Rv] + [Uv[Rk] for Uk, Uv in U.items()] for Rk, Rv in R.items()] + [[0] + [-Uk[0] for Uk, Uv in U.items()]]
np.array(Xs)
simplex_max(Xs, rows=['Z', 'D', 'K', 'F'], columns=['free', 'X1', 'X2', 'X3'])
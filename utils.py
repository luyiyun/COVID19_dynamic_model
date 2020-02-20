from datetime import date
import pickle
import json

import numpy as np
from scipy import sparse


def clear_time(times):
    """ 将时间从字符串变为date实例 """
    need_times = [date.fromisoformat(t.strip()) for t in times]
    return need_times


def clear_value(values):
    """ 将百分比字符串变成float """
    return [float(v.strip()[:-1]) for v in values]


def difference2(A, B):
    common = set(A).intersection(set(B))
    A_rest = list(set(A).difference(common))
    B_rest = list(set(B).difference(common))
    return list(common), [A_rest, B_rest]


def df_to_mat(df, shape, source="source", target="target", values="value"):
    """ 将稀疏格式变成密集矩阵 """
    smat = sparse.coo_matrix(
        (df[values].values, (df[source].values, df[target].values)),
        shape=shape
    )
    return np.array(smat.todense())  # np.array将matrix变成array，不然ode会出错


def normalize(mat):
    """ 归一化mat，使之每一行和为1，没用到 """
    return mat / mat.sum(axis=1, keepdims=True)


def time_str2ord(t):
    return date.fromisoformat(t).toordinal()


def time_date2diff(t, t0):
    return t.toordinal() - time_str2ord(t0)


def time_str2diff(t, t0=None):
    t_ord = time_str2ord(t)
    t0_ord = time_str2ord(t0)
    return t_ord - t0_ord


def save(obj, filename, type="pkl"):
    if type == "pkl":
        with open(filename, "wb") as f:
            pickle.dump(obj, f)
    elif type == "json":
        with open(filename, "w") as f:
            json.dump(obj, f)
    else:
        raise NotImplementedError


def load(filename, type="pkl"):
    if type == "pkl":
        with open(filename, "rb") as f:
            obj = pickle.load(f)
    elif type == "json":
        with open(filename, "r") as f:
            obj = json.load(f)
    else:
        raise NotImplementedError
    return obj

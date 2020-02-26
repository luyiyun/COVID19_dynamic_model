import os
from datetime import date
import pickle
import json
import re

import numpy as np
from scipy import sparse
from argparse import ArgumentParser


ITER_COUNT = 0


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


def callback(x, f, context):
    global ITER_COUNT
    print("第%d次迭代，当前最小值%.4f，当前状态%d" % (ITER_COUNT, f, context))
    ITER_COUNT += 1


def protect_decay1(t, t0, tm, eta, epsilon=0.001):
    """
    表示控制措施，实际上就是一个值域在0-1之间的函数，其乘到传染率系数上去。
    t0表示措施开始的时间；
    eta表示我们最终希望传染率降低多少；
    tm表示基本达到我们的目标时的所需的总时间；
    epsilon是一个数值安全参数；

    注意，我们这里对于所有的地区使用相同的函数，则我们后面的t0、eta、tm使用
    scalar即可，则得到的结果也是scalar，表示对于所有的地区使用相同的控制；
    但我们也可以将这些参数使用向量，则得到的结果也是向量，表示对于不同的地区使用不同的措施
    """
    lam = np.log((1 - epsilon) / epsilon) / tm * 2
    res = eta / (1 + np.exp(lam * (t - t0 - tm / 2))) + 1 - eta
    bool = t > t0
    return bool * res + (1 - bool)


def protect_decay2(t, t0, k):
    le_bool = t <= t0
    res = le_bool + (1 - le_bool) * np.exp(-k * (t - t0))
    return res


class PmnFunc:
    """
    得到Pmn关于t的函数。即得到每个时间点上的人口流动比例。
    """
    def __init__(self, pmn, use_mean=False):
        """
        pmn是一个dict，其key是date属性，记录的是哪一天，其value是一个ndarray的
        matrix，其第mn个元素表示的是第m个地区迁徙到第n个地区的人口占所有迁出m地区人口
        的比例
        """
        self.pmn = pmn
        self.use_mean = use_mean
        if self.use_mean:
            vv, counts = 0, 0
            for v in self.pmn.values():
                vv += v
                counts += 1
            self.pmn_mean = vv / counts
        else:
            self.pmn_times = list(self.pmn.keys())
            self.pmn_times_min, self.pmn_times_max = \
                min(self.pmn_times), max(self.pmn_times)

    def __call__(self, ord_time):
        """
        ord_time，相对于t0的相对计时，比如，t0那一天就是0，其后一天是1.
        """
        if self.use_mean:
            return self.pmn_mean
        if ord_time <= self.pmn_times_min:
            return self.pmn[self.pmn_times_min]
        elif ord_time >= self.pmn_times_max:
            return self.pmn[self.pmn_times_max]
        else:
            ord_time_int = int(ord_time)
            diff = self.pmn[ord_time_int+1] - self.pmn[ord_time_int]
            result = self.pmn[ord_time_int] + diff * (ord_time - ord_time_int)
        return result


class PmnFunc2:
    """
    得到Pmn关于t的函数。即得到每个时间点上的人口流动比例。
    """
    def __init__(self, pmn, use_mean=False):
        """
        只考虑湖北的人口迁出
        """
        self.pmn = pmn
        self.use_mean = use_mean
        if self.use_mean:
            vv, counts = 0, 0
            for v in self.pmn.values():
                vv += v
                counts += 1
            self.pmn_mean = vv / counts
        else:
            self.pmn_times = list(self.pmn.keys())
            self.pmn_times_min, self.pmn_times_max = \
                min(self.pmn_times), max(self.pmn_times)

    def __call__(self, ord_time):
        """
        ord_time，相对于t0的相对计时，比如，t0那一天就是0，其后一天是1.
        """
        if self.use_mean:
            return self.pmn_mean
        if ord_time <= self.pmn_times_min:
            return self.pmn[self.pmn_times_min]
        elif ord_time >= self.pmn_times_max:
            return self.pmn[self.pmn_times_max]
        else:
            ord_time_int = int(ord_time)
            diff = self.pmn[ord_time_int+1] - self.pmn[ord_time_int]
            result = self.pmn[ord_time_int] + diff * (ord_time - ord_time_int)
        return result


class GammaFunc1:
    def __init__(self, gammas, use_mean=False):
        """
        使用真实的人口迁出比数据（来自百度，但是实际效果不好）

        描述不同地区的人口迁出比的函数，输入时间（相对于t0的相对时间），输出当前时间的各个地区
        的人口迁出比，是一个向量。
            :param gammas: dict，key是相对时间点，value是对应的ndarray，表示每个地区的
                在该时间点上的人口迁出比。
            :param use_mean=False: 如果是True，则使用一个固定的值为人口迁出比，即当前
                所有时间点上的人口迁出比的均值。
        """
        self.gammas = gammas
        self.use_mean = use_mean
        if use_mean:
            vv, counts = 0, 0
            for v in self.gammas.values():
                vv += v
                counts += 1
            self.gamma_mean = vv / counts
        else:
            self.gamma_times = list(self.gammas.keys())
            self.gamma_times_min, self.gamma_times_max = \
                min(self.gamma_times), max(self.gamma_times)

    def __call__(self, ord_time):
        if self.use_mean:
            return self.gamma_mean
        if ord_time <= self.gamma_times_min:
            return self.gammas[self.gamma_times_min]
        elif ord_time >= self.gamma_times_max:
            return self.gammas[self.gamma_times_max]
        else:
            ord_time_int = int(ord_time)
            diff = self.gammas[ord_time_int+1] - self.gammas[ord_time_int]
            result = self.gammas[ord_time_int] + \
                diff * (ord_time - ord_time_int)
        return result


class GammaFunc2:
    def __init__(self, gammas=0.1255, protect_t0=None):
        """
        只使用恒定的、全国相同的人口迁出比，在protect_t0之后会立即或逐渐降为0

        描述不同地区的人口迁出比的函数，输入时间（相对于t0的相对时间），输出当前时间的各个地区
        的人口迁出比，是一个向量。
            :param gammas: float，表示全国平均的人口迁出比，所以地区使用一个值。
            :param protect_t0: float或ndarray，是每个地区的预防措施时间，在此时间后人
                口流动会线性下降至0，其下降的斜率是ks。如果是None，则认为会严格按照给定的
                gammas进行人口流动，不会有线性下降到0的过程。
            :param ks: float或ndarray，表示每个地区限制人流的措施的实施程度，即上面
                线性下降过程的斜率。如果是None，则表示在t0时刻直接降为0
        """
        self.gammas = gammas
        self.protect_t0 = protect_t0

    def __call__(self, ord_time):
        """ 相对计时，相对于start day的计时，在start_day=0 """
        if self.protect_t0 is not None:
            le_bool = ord_time <= self.protect_t0
            return self.gammas * le_bool
        return self.gammas


class MyArguments(ArgumentParser):
    """ 为了方便，把一些共用的参数放在一起，并集中处理一下"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("--save_dir")
        self.add_argument("--region_type", default="province",
                          choices=("city", "province"))
        self.add_argument(
            "--model", default="fit",
            help=("默认是None，即不使用训练的模型，而是直接使用命令行赋予的参数, "
                  "不然则读取拟合的参数，命令行赋予的参数无效, 如果是fit，"
                  "则取进行自动的拟合")
        )
        self.add_argument(
            "--regions", default=None, nargs="+",
            help=("默认是None，则对于省份或城市都使用不同的默认值，不然，则需要键入需要估"
                  "计的地区名。如果是all，则将所有的都计算一下看看")
        )
        self.add_argument("--t0", default="2019-12-31", help="疫情开始的那天")
        self.add_argument("--y0", default=100, type=float,
                          help="武汉或湖北在t0那天的感染人数")
        self.add_argument("--tm", default="2020-03-31", help="需要预测到哪天")
        self.add_argument("--protect_args", default=[0], type=float, nargs="+")
        self.add_argument("--fit_score", default="nll", choices=["nll", "mse"])
        self.add_argument("--fit_time_start", default="2020-02-01")
        self.add_argument("--use_whhb", action="store_true")
        self.add_argument("--fit_method", default="geatpy",
                          choices=["geatpy", "annealing"])

    def parse_args(self):
        args = super().parse_args()
        args.save_dir = os.path.join("./RESULTS2/", args.save_dir)
        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
        if args.regions is None:
            if args.region_type == "city":
                args.regions = [
                    "武汉", "孝感", "荆州", "荆门", "随州", "黄石", "宜昌",
                    "鄂州", "北京", "上海", "哈尔滨", "淄博"
                ]
            else:
                args.regions = ["湖北", "北京", "上海", "广东", "湖南",
                                "浙江", "河南", "山东", "黑龙江"]
        # 将时间都处理成相对于t0的相对时间
        args.tm_relative = time_str2diff(args.tm, args.t0)
        args.fit_start_relative = time_str2diff(args.fit_time_start, args.t0)
        return args


def parser_key(key):
    if "[" in key and "]" in key:
        key1, key2 = key.split("[")
        key2 = key2[:-1]  # 去掉]
        if ":" in key2:
            ind1, ind2 = key2.split(":")
            slice_ind = slice(int(ind1), int(ind2))
        else:
            slice_ind = int(key2)
    else:
        key1, slice_ind = key, None

    if "-" in key1:
        key11, key22 = key1.split("-")
    else:
        key11, key22 = key1, None

    return key11, key22, slice_ind

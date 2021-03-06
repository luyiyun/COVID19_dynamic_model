import os
from datetime import date, timedelta
import pickle
import json
from collections.abc import Sequence

import numpy as np
from scipy import sparse
from argparse import ArgumentParser


ITER_COUNT = 0


""" ========== 辅助函数 ========== """


def df_to_mat(df, shape, source="source", target="target", values="value"):
    """
    将稀疏格式变成密集矩阵

    Arguments:
        df {DataFrame} -- 储存坐标和值的df
        shape {2-tuple} -- (nrow, ncol)

    Keyword Arguments:
        source {str} -- df中表示行坐标的列名 (default: {"source"})
        target {str} -- df中表示列坐标的列名 (default: {"target"})
        values {str} -- df中表示值的列名 (default: {"value"})

    Returns:
        ndarray -- shape=shape, matrix
    """
    smat = sparse.coo_matrix(
        (df[values].values, (df[source].values, df[target].values)),
        shape=shape
    )
    return np.array(smat.todense())  # np.array将matrix变成array，不然ode会出错


def correct(arr, ill_ind, period=None):
    if period is not None:
        t1 = ill_ind - period
    else:
        t1 = 0
    arr_part = arr[t1:ill_ind]
    arr_part = arr_part / arr_part.sum()
    add_value = arr_part * (arr[ill_ind] - arr[ill_ind-1])
    arr_part = arr_part + add_value
    res = arr.copy()
    res[t1:ill_ind] = res[t1:ill_ind] + arr_part
    return res


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


""" ========== 时间相关 ========== """


# def clear_time(times):
#     """
#     将xxxx-xx-xx格式转换成Date对象

#     Arguments:
#         times {list of str} -- xxxx-xx-xx字符串组成的列表

#     Returns:
#         list of Date -- Date组成的列表
#     """
#     need_times = [date.fromisoformat(t.strip()) for t in times]
#     return need_times


# def clear_value(values):
#     """
#     xx% --> float(xx)

#     Arguments:
#         values {list of str} -- xx%组成的字符串列表

#     Returns:
#         list of float -- float列表
#     """
#     """ 将百分比字符串变成float """
#     return [float(v.strip()[:-1]) for v in values]


# def time_str2ord(t):
#     """
#     xxxx-xx-xx --> ordinal format

#     Arguments:
#         t {str} -- xxxx-xx-xx format

#     Returns:
#         int -- ordinal format
#     """
#     return date.fromisoformat(t).toordinal()


# def time_date2diff(t, t0):
#     """
#     Date对象 --> 相对于t0过了多少天

#     Arguments:
#         t {Date} -- Date对象
#         t0 {str} -- xxxx-xx-xx format, 起始时间

#     Returns:
#         int -- 相对时间
#     """
#     return t.toordinal() - time_str2ord(t0)


# def time_str2diff(t, t0=None):
#     """
#     xxxx-xx-xx --> 相对于t0过了多少天

#     Arguments:
#         t {str} -- xxxx-xx-xx format
#         t0 {str} -- xxxx-xx-xx format, 起始时间

#     Returns:
#         int -- 相对时间
#     """
#     t_ord = time_str2ord(t)
#     t0_ord = time_str2ord(t0)
#     return t_ord - t0_ord


# class Time:
#     _threshold = date.fromisoformat("2000-01-01").toordinal()

#     def __init__(self, tim, t0):
#         self.t0 = t0
#         self.t0_ord = time_str2ord(t0)
#         if isinstance(tim, (tuple, list, np.ndarray)):
#             test_elem = tim[0]
#             self.isSeq = True
#         else:
#             test_elem = tim
#             self.isSeq = False
#         if isinstance(test_elem, str):
#             self.init_from_str(tim)
#         elif test_elem > self._threshold:
#             self.init_from_ord(tim)
#         else:
#             self.init_from_rel(tim)

#     def init_from_str(self, strfmt):
#         self.str = strfmt
#         if self.isSeq:
#             self.ord = np.array([time_str2ord(s) for s in strfmt])
#             self.relative = np.array([
#                 time_str2diff(s, self.t0) for s in strfmt])
#         else:
#             self.ord = time_str2ord(strfmt)
#             self.relative = time_str2diff(strfmt, self.t0)

#     def init_from_ord(self, ord):
#         self.ord = ord
#         if self.isSeq:
#             self.str = [str(date.fromordinal(o)) for o in ord]
#             self.relative = np.array([o - self.t0_ord for o in ord])
#         else:
#             self.str = str(date.fromordinal(ord))
#             self.relative = ord - self.t0_ord

#     def init_from_rel(self, rel):
#         self.relative = rel
#         if self.isSeq:
#             self.ord = np.array([r + self.t0_ord for r in rel])
#             self.str = [str(date.fromordinal(o)) for o in self.ord]
#         else:
#             self.ord = rel + self.t0_ord
#             self.str = str(date.fromordinal(self.ord))


class CustomDate:
    """
    用来表示日期的类，用来表示单个日期。
    - 主要功能有：
        1. 实例化时，增加了t0参数，即表示起始时间。如果此参数为None，没有影响，如果此参数
            不为None，则会另外计算一个delta属性，为当前时间点到t0的距离（int）；
        2. 转换为ord格式（ord）、转换为str格式（str（d）,str）；
        3. 可以进行+（int或timedelta，返回CustomDate）、-（
            int或timedelta，返回CustomDate；CustomDate或date则返回int）
    """
    def __init__(self, t, t0=None):
        self.t, self.t0 = t, t0
        if isinstance(t, date):
            self._t = t
        elif isinstance(t, str):
            self._t = date.fromisoformat(t)
        elif isinstance(t, int):
            self._t = date.fromordinal(t)
        elif isinstance(t, (float, np.number)):
            self._t = date.fromordinal(int(t))
        else:
            raise NotImplementedError
        if t0 is not None:
            if isinstance(t0, self.__class__):
                self._t0 = t0._t
            elif isinstance(t0, str):
                self._t0 = date.fromisoformat(t0)
            elif isinstance(t0, date):
                self._t0 = t0
            elif isinstance(t, int):
                self._t0 = date.fromordinal(t0)
            elif isinstance(t, (float, np.number)):
                self._t0 = date.fromordinal(int(t0))
            else:
                raise NotImplementedError
            self._timedelta = (self._t - self._t0).days
        else:
            self._timedelta = None

    def __add__(self, other):
        """ 返回的是CustomDate对象或CustomDates对象 """
        if isinstance(other, timedelta):
            return self.__class__(self._t + other, self.t0)
        elif isinstance(other, int):
            return self.__add__(timedelta(other))
        elif isinstance(other, (float, np.number)):
            return self.__add__(int(other))
        elif isinstance(other, (list, np.ndarray)):
            return CustomDates([self.__add__(i) for i in other])
        else:
            raise ValueError(
                "A + B, B must be CustomDate, timedelta or its seq.")

    def __sub__(self, other):
        """
        如果减去的是CustomDate或date对象，返回的是int；
        如果减去的是int或timedelta，则返回CustomDate对象；
        """
        if isinstance(other, date):
            return (self._t - other).days
        elif isinstance(other, self.__class__):
            return self.__sub__(other._t)
        elif isinstance(other, timedelta):
            return self.__class__(self._t - other, self.t0)
        elif isinstance(other, int):
            return self.__sub__(timedelta(days=other))
        elif isinstance(other, (float, np.number)):
            return self.__sub__(int(other))
        else:
            raise ValueError(
                "A - B, B must be one of CustomDate, date, int and timedelta.")

    def __str__(self):
        return str(self._t)

    def __repr__(self):
        return "%s, delta: %s" % (self.__str__(), str(self.delta))

    @property
    def ord(self):
        return np.float(self._t.toordinal())

    @property
    def str(self):
        return str(self._t)

    @property
    def delta(self):
        return np.nan if self._timedelta is None else np.float(self._timedelta)

    @staticmethod
    def from_delta(delta, t0):
        return CustomDate(t0, t0) + delta


class CustomDates(Sequence):
    """
    上面那个类的复数版本，支持整体的+/-
    """
    def __init__(self, ts):
        super().__init__()
        for t in ts:
            if not isinstance(t, CustomDate):
                raise ValueError("element of ts must be CustomDate.")
        self._ts = ts

    def __len__(self):
        return len(self._ts)

    def __getitem__(self, index):
        return self._ts[index]

    def __add__(self, other):
        if isinstance(other, (list, np.ndarray)):
            assert self.__len__() == len(other)
            res = self.__class__([t1 + t2 for t1, t2 in zip(self._ts, other)])
        else:
            res = self.__class__([t + other for t in self._ts])
        if all([isinstance(r, CustomDate) for r in res]):
            return self.__class__(res)
        elif all([isinstance(r, (int, None)) for r in res]):
            return np.array(res).astype("float")
        else:
            return res

    def __sub__(self, other):
        if isinstance(other, (list, np.ndarray, CustomDates)):
            assert self.__len__() == len(other)
            res = [t1 - t2 for t1, t2 in zip(self._ts, other)]
        else:
            res = [t - other for t in self._ts]
        if all([isinstance(r, CustomDate) for r in res]):
            return self.__class__(res)
        elif all([isinstance(r, (int, None)) for r in res]):
            return np.array(res).astype("float")
        else:
            return res

    def __str__(self):
        return [str(t) for t in self._ts]

    def __repr__(self):
        return "\n".join([t.__repr__() for t in self._ts])

    @property
    def str(self):
        return self.__str__()

    @property
    def ord(self):
        return np.array([t.ord for t in self._ts])

    @property
    def delta(self):
        return np.array([t.delta for t in self._ts])

    @staticmethod
    def from_range(start, stop, t0=None):
        """ 这里得到的CustomDates是包含stop那一天的 """
        if isinstance(start, str):
            start = date.fromisoformat(start).toordinal()
        elif not isinstance(start, int):
            raise ValueError("start must be ordinal or xxxx-xx-xx.")
        if isinstance(stop, str):
            stop = date.fromisoformat(stop).toordinal()
        elif not isinstance(stop, int):
            raise ValueError("stop must be ordinal or xxxx-xx-xx.")
        return CustomDates([
            CustomDate(i, t0)
            for i in range(start, stop+1)
        ])


""" ========== 回调函数，用于annealing fit方法 ========== """


def callback(x, f, context):
    global ITER_COUNT
    print("第%d次迭代，当前最小值%.4f，当前状态%d" % (ITER_COUNT, f, context))
    ITER_COUNT += 1


""" ========== 控制措施函数 ========== """


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


""" ========== 人口流动函数 ========== """


class PmnFunc:
    """
    得到Pmn关于t的函数。即得到每个时间点上的人口流动比例。
    """
    def __init__(self, pmn):
        """
        pmn是一个dict，其key是time delta，记录的是哪一天，其value是一个ndarray的
        matrix，其第mn个元素表示的是第m个地区迁徙到第n个地区的人口占所有迁出m地区人口
        的比例
        """
        self.pmn = pmn
        self.pmn_times = list(self.pmn.keys())
        self.pmn_times_min, self.pmn_times_max = \
            min(self.pmn_times), max(self.pmn_times)

    def __call__(self, ord_time):
        """
        ord_time，相对于t0的相对计时，比如，t0那一天就是0，其后一天是1.
        """
        if ord_time <= self.pmn_times_min:
            return self.pmn[self.pmn_times_min]
        elif ord_time >= self.pmn_times_max:
            return self.pmn[self.pmn_times_max]
        else:
            ord_time_int = int(ord_time)
            diff = self.pmn[ord_time_int+1] - self.pmn[ord_time_int]
            result = self.pmn[ord_time_int] + diff * (ord_time - ord_time_int)
        return result


""" ========== 人口迁徙比函数 ========== """


class GammaFunc1:
    def __init__(self, gammas, zero_period=None):
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
        self.zero_period = zero_period
        self.gamma_times = list(self.gammas.keys())
        self.gamma_times_min, self.gamma_times_max = \
            min(self.gamma_times), max(self.gamma_times)

    def __call__(self, ord_time):
        multi_value = 1
        if self.zero_period is not None:
            if (
                ord_time >= self.zero_period[0] and
                ord_time <= self.zero_period[1]
            ):
                multi_value = 0
        if ord_time <= self.gamma_times_min:
            return self.gammas[self.gamma_times_min] * multi_value
        elif ord_time >= self.gamma_times_max:
            return self.gammas[self.gamma_times_max] * multi_value
        else:
            ord_time_int = int(ord_time)
            diff = self.gammas[ord_time_int+1] - self.gammas[ord_time_int]
            result = self.gammas[ord_time_int] + \
                diff * (ord_time - ord_time_int)
        return result * multi_value


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


""" ========== 导入数据和数据预处理 ========== """


class MyArguments(ArgumentParser):
    """ 为了方便，把一些共用的参数放在一起，并集中处理一下"""
    def __init__(self, *args, save_root_dir="./RESULTS2/", **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("--save_dir")
        # self.add_argument("--region_type", default="province",
        #                   choices=("city", "province"))
        self.add_argument(
            "--fit", action="store_true",
            help="如果使用此参数，则会使用遗传算法寻找参数"
        )
        self.add_argument(
            "--model", default=None,
            help=("默认是None，即不使用训练的模型，而是直接使用命令行赋予的参数, "
                  "不然则读取拟合的参数，命令行赋予的参数无效")
        )
        self.add_argument("--t0", default="2019-12-01", help="疫情开始的那天")
        self.add_argument("--y0", default=1, type=float,
                          help="武汉或湖北在t0那天的感染人数")
        self.add_argument("--tm", default="2020-04-30", help="需要预测到哪天")
        self.add_argument("--fit_score", default="nll",
                          choices=["nll", "mse", "mae"])
        self.add_argument("--fit_time_start", default="2020-02-01")
        self.add_argument("--mask", action="store_true")
        self.add_argument("--fit_method", default="SEGA",
                          choices=[
                              "SEGA", "multiSEGA", "psySEGA", "annealing",
                              "rand1lDE"
                            ]
                          )
        self.add_argument("--geatpy_maxgen", default=25, type=int)
        self.add_argument("--geatpy_nind", default=500, type=int)
        self.add_argument(
            "--geatpy_npop", default=5, type=int,
            help="种群数量, 只有在使用multiSEGA时才有效"
        )

        self.save_root_dir = save_root_dir

    def parse_args(self):
        args = super().parse_args()
        args.save_dir = os.path.join(self.save_root_dir, args.save_dir)
        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
        return args


class Dataset:

    def __init__(self, filename, t0, tm, fit_start_t):

        dats = load(filename, "pkl")

        # 重要时间点
        self.t0 = CustomDate(t0, t0)               # 疫情开始时间
        self.protect_t0 = CustomDates([            # 防控开始时间，即开始响应的时间
            CustomDate(d, t0)
            for d in dats["response_time"]
        ])
        self.epi_t0 = CustomDate(dats["epidemic_t0"], t0)    # 第一例疫情确诊时间
        self.tm = CustomDate(tm, t0)                         # 预测结束时间
        self.fit_start_t = CustomDate(fit_start_t, t0)       # 使用的数据的开始时间
        # 重要时间段
        self.epi_times = self.epi_t0 + \
            np.arange(dats["epi_times"])                     # 确诊病例时间段
        self.pred_times = CustomDates.from_range(t0, tm, t0)   # 预测时间段
        self.out20_times = CustomDate(dats["out_trend_t0"], t0) +\
            np.arange(dats["out_trend20"].shape[0])            # 迁出人口比时间段
        self.out19_times = CustomDate(dats["out_trend_t0"], t0) +\
            np.arange(dats["out_trend19"].shape[0])       # 迁出人口比时间段
        self.zero_period = CustomDates.from_range(        # 春节，人口流动可以设为0
            "2020-01-25", "2020-02-01", t0=t0
        )
        # 随时间变化的取值，date delta为key的dict来表示
        self.pmn_matrix_relative = {
            CustomDate(k, t0).delta: v for k, v in dats["pmn"].items()}
        self.out20_dict = {}
        for i, t in enumerate(self.out20_times.delta):
            self.out20_dict[t] = dats["out_trend20"][i, :]
        self.out19_dict = {}
        for i, t in enumerate(self.out19_times.delta):
            self.out19_dict[t] = dats["out_trend19"][i, :]
        # 其他
        self.regions = dats["regions"]
        self.populations = dats["population"]
        self.trueH, self.trueR, self.trueD = dats["trueH"], dats["trueR"], \
            dats["trueD"]
        self.num_regions = len(self.regions)

        # 这里对山东和湖北的数据进行一下修正
        # hb_ind, sd_ind = 0, self.regions.index("山东")
        # import ipdb; ipdb.set_trace()
        # self.trueH[:, hb_ind] = correct(self.trueH[:, hb_ind], 20, 12)
        # self.trueH[:, sd_ind] = correct(self.trueH[:, sd_ind], 26, 12)


if __name__ == "__main__":
    """ test CustomDate and CustomDates """
    if False:
        a = CustomDate("2020-01-01", "2019-12-01")
        b = CustomDate("2020-01-10")
        A = CustomDates([a, b])

        print((a + np.arange(0, 10)).str)
        print((a + np.arange(0, 10)).delta)

        print(A.str)
        print(A.ord)
        print(A.delta)

        print(A - a)
        print(A - date.fromisoformat("2020-01-01"))
        print((A - 10).str)
        print((A - timedelta(10)).str)

        print((A + 10).str)
        print((A + timedelta(10)).str)

        for i in A:
            print(i.str)

        B = CustomDates([
            CustomDate("2019-01-15"),
            CustomDate("2019-01-16"),
        ])
        C = [10, 11]
        D = [
            date.fromisoformat("2019-01-15"),
            date.fromisoformat("2019-01-16")
        ]
        E = [timedelta(10), timedelta(11)]
        F = [10, 11, 12]

        print(A - B)
        print((A - B).__class__)
        print((A - C).str)
        print(A - D)
        print((A - E).str)
        print(A - F)
    if True:
        dat = Dataset(
            "./DATA/Provinces.pkl", "2019-12-01", "2020-04-30", "2020-02-01"
        )

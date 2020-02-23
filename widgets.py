
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


class GammaFunc2:
    def __init__(self, gammas, use_mean=False):
        """ 所有城市移动人口占全国总人口的比例，暂时设为一个常数值 """
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
        """ 相对计时，相对于start day的计时，在start_day=0 """
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


class GammaFunc:
    def __init__(self, protect_t0):
        """ 所有城市移动人口占全国总人口的比例，暂时设为一个常数值 """
        self.protect_t0 = protect_t0

    def __call__(self, ord_time):
        """ 相对计时，相对于start day的计时，在start_day=0 """
        le_bool = ord_time <= self.protect_t0
        return 0.1255 * le_bool

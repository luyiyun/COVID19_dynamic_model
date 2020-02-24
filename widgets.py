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


class GammaFunc1:
    def __init__(self, gammas, use_mean=False):
        """
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
    def __init__(self, gammas=0.1255, protect_t0=None, ks=None):
        """
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
        self.ks = ks

    def __call__(self, ord_time):
        """ 相对计时，相对于start day的计时，在start_day=0 """
        if self.protect_t0 is not None:
            le_bool = ord_time <= self.protect_t0
            now_gammas = self.gammas * le_bool
            if self.ks is not None:
                now_gammas += (self.gammas-self.ks*(ord-self.protect_t0)) *\
                    (1 - le_bool)
                return now_gammas * (now_gammas > 0)
            return now_gammas
        return self.gammas

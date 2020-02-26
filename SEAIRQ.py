from copy import deepcopy

import numpy as np

from model import InfectiousBase


class SEAIRQ(InfectiousBase):
    def __init__(
        self, populations, t0, y0for1, protect, protect_args,
        De=5, Dq=14, c=13.0046, beta=2.03e-9, q=1.88e-7,
        rho=0.6834, deltaI=0.1328, deltaQ=0.1259, gammaI=0.1029, gammaA=0.2978,
        gammaH=0.1024, alpha=0.0009, theta=1.6003, nu=1.5008, score_type="mse"
    ):
        """
        y0 = [H R D E A I Sq Eq] + [S]
        """
        super().__init__(
            populations, t0, y0for1, protect, protect_args,
            De, Dq, c, beta, q, rho, deltaI, deltaQ, gammaI,
            gammaA, gammaH, alpha, theta, nu, score_type
        )
        self.sigma = 1 / De
        self.lam = 1 / Dq

        # 计算y0
        y0s_remain = populations - np.sum(y0for1)
        y0s = list(y0for1) + [y0s_remain]
        self.y0 = np.array(y0s) / populations

    def __call__(self, t, ALL):
        """
        t是时间参数，
        y0 = [H R D E A I Sq Eq] + [S]
        """
        HH = ALL[0]
        # RR = ALL[self.num_regions:2*self.num_regions]
        # DD = ALL[2*self.num_regions:3*self.num_regions]
        EE = ALL[3]
        AA = ALL[4]
        II = ALL[5]
        SSq = ALL[6]
        EEq = ALL[7]
        SS = ALL[8]

        # 如果有保护措施，这里计算受到保护措施影响的c和q
        if self.protect:
            decayc, decayq = self.protect_decay(t, **self.protect_args)
            ci, qi = self.c * decayc, self.q * decayq
        else:
            ci, qi = self.c, self.q

        # 计算导数
        SIAE = SS * (II + self.theta * AA + self.nu * EE)
        HH_ = self.deltaI*II+self.deltaQ*EEq-(self.alpha+self.gammaH)*HH
        RR_ = self.gammaI*II+self.gammaA*AA+self.gammaH*HH
        DD_ = self.alpha * (II + HH)
        EE_ = self.beta*ci*(1-qi)*SIAE-self.sigma*EE
        AA_ = self.sigma*(1-self.rho)*EE-self.gammaA*AA
        II_ = self.sigma*self.rho*EE-(self.deltaI+self.alpha+self.gammaI)*II
        SSq_ = (1-self.beta)*ci*qi*SIAE-self.lam*SSq
        EEq_ = self.beta*ci*qi*SIAE-self.deltaQ*EEq
        SS_ = -(self.beta*ci+ci*qi*(1-self.beta))*SIAE+self.lam*SSq

        output = np.r_[HH_, RR_, DD_, EE_, AA_, II_, SSq_, EEq_, SS_]
        return output

    def predict(self, times):
        ALL = super().predict(times)
        HH = ALL[:, 0] * self.populations
        RR = ALL[:, 1] * self.populations
        DD = ALL[:, 2] * self.populations
        EE = ALL[:, 3] * self.populations
        AA = ALL[:, 4] * self.populations
        II = ALL[:, 5] * self.populations
        SSq = ALL[:, 6] * self.populations
        EEq = ALL[:, 7] * self.populations
        SS = ALL[:, 8] * self.populations
        return HH, RR, DD, EE, AA, II, SSq, EEq, SS

    @staticmethod
    def protect_decay(t, t0, c_k, q_k):
        # le_bool = t <= t0
        # decayc = le_bool + (1 - le_bool) * np.exp(-c_k * (t - t0))
        # decayq = le_bool + (1 - le_bool) * np.exp(-q_k * (t - t0))
        return decayc, decayq

    def score(self, times, trueH, trueR, trueD, mask=None):
        """
        因为这里是多个地区的预测，所以true_infects也是一个二维矩阵，即
        shape = num_times x num_regions
        """
        predH, predR, predD = self.predict(times)[:3]
        if self.score_type == "mse":
            diff = (predH - trueH) ** 2 + \
                (predR - trueR) ** 2 + \
                (predD - trueD) ** 2
        elif self.score_type == "nll":
            diff = (predH - np.log(predH) * trueH) + \
                (predR - np.log(predR) * trueR) + \
                (predD - np.log(predD) * trueD)
        else:
            raise ValueError
        if mask is not None:
            diff = diff * mask
        return np.mean(diff)

    def R0(self, ts):
        part1 = self.beta * self.c * self.rho * (1 - self.q) / \
            (self.deltaI + self.alpha + self.gammaI)
        part2 = self.beta * self.c * (1 - self.rho) * (1 - self.q) * \
            self.theta / self.gammaA
        part3 = self.beta * self.c * self.nu * (1 - self.q) / self.sigma
        return (part1 + part2 + part3) * self.populations

    def set_params(self, params):
        """
        y0 = [H R D E A I Sq Eq] + [S]
        比较早的时间，H、R、D、Sq、Eq都是0，EAI为需要拟合的参数
        """
        update_args = {
            # 各地区都一致的参数
            "c": params[0], "beta": params[1], "q": params[2],
            "rho": params[3], "deltaI": params[4], "deltaQ": params[5],
            "gammaI": params[6], "gammaA": params[7], "gammaH": params[8],
            "alpha": params[9], "theta": params[10], "nu": params[11],
        }
        self.kwargs.update(update_args)
        # self.kwargs["y0for1"][3:6] = params[12:15]
        # self.kwargs["y0for1"][-1] = params[15]
        self.kwargs["protect_args"]["c_k"] = params[12]
        self.kwargs["protect_args"]["q_k"] = params[13]
        self.kwargs["y0for1"][3:6] = params[14:17]

        self.__init__(**self.kwargs)

    def params_fit_range(self):
        """
        y0 = [H R D E A I Sq Eq] + [S]
        """
        num_params = 17
        lb = np.zeros(num_params)
        ub = np.r_[
            100, 1, 1, 1, 1, 1, 1, 1, 1, 1, 10, 10,  # 12个恒定参数
            0.5, 0.5,  # c_k和q_k
            100, 100, 100,  # 3个初值参数
        ]
        return num_params, lb, ub



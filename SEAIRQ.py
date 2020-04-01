from copy import deepcopy
from collections import OrderedDict

import numpy as np

from model import InfectiousBase
import utils


class SEAIRQ(InfectiousBase):
    def __init__(
        self, populations, y0for1, protect, protect_args,
        De=5, Dq=14, c=13.0046, beta=2.03e-9, q=1.88e-7,
        rho=0.6834, deltaI=0.1328, deltaQ=0.1259, gammaI=0.1029, gammaA=0.2978,
        gammaH=0.1024, alpha=0.0009, theta=1.6003, nu=1.5008, score_type="mse"
    ):
        """ y0 = [H R D E A I Sq Eq] + [S] """
        super().__init__(
            populations, y0for1, protect, protect_args,
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
        """ y0 = [H R D E A I Sq Eq] + [S] """
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
        decayc = utils.protect_decay2(t, t0, c_k)
        decayq = utils.protect_decay2(t, t0, q_k)
        return decayc, decayq

    def score(self, times, trueH, trueR, trueD, mask=None):
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

    @property
    def fit_params_info(self):
        params = OrderedDict()
        params["c"] = (1, 0, 100)
        params["beta"] = (1, 0, 1)
        params["q"] = (1, 0, 1)
        params["rho"] = (1, 0, 1)
        params["deltaI"] = (1, 0, 1)
        params["deltaQ"] = (1, 0, 1)
        params["gammaI"] = (1, 0, 1)
        params["gammaA"] = (1, 0, 1)
        params["gammaH"] = (1, 0, 1)
        params["alpha"] = (1, 0, 1)
        params["theta"] = (1, 0, 10)
        params["nu"] = (1, 0, 10)
        params["protect_args-c_k"] = (1, 0, 1)
        params["protect_args-q_k"] = (1, 0, 1)
        params["y0for1[3:6]"] = (3, 0, 100)
        return params


def main():
    """ 命令行参数及其整理 """
    parser = utils.MyArguments()
    parser.add_argument("--De", default=5, type=float)
    parser.add_argument("--Dq", default=14, type=float)
    args = parser.parse_args()  # 对于一些通用的参数，这里已经进行整理了
    # 保存args到路径中
    utils.save(args.__dict__, os.path.join(args.save_dir, "args.json"), "json")

    """ 读取准备好的数据 """
    if args.region_type == "city":
        dat_file = "./DATA/City.pkl"
    else:
        dat_file = "./DATA/Provinces.pkl"
    dats = utils.load(dat_file, "pkl")

    # 时间调整(我们记录的数据都是以ordinal格式记录，但输入模型中需要以相对格式输入,其中t0是0)
    t0 = utils.time_str2ord(args.t0)                              # 疫情开始时间
    protect_t0_relative = dats["response_time"] - t0              # 防控开始时间
    # protect_t0_relative = utils.time_str2diff("2020-01-23", args.t0)
    epi_t0_relative = dats["epidemic_t0"] - t0                    # 第一确诊时间
    pmn_matrix_relative = {                                       # pmn的时间
        (k-t0): v for k, v in dats["pmn"].items()}
    epi_times_relative = np.arange(                               # 确诊病例时间段
        epi_t0_relative, epi_t0_relative + dats["trueH"].shape[0]
    )
    pred_times_relative = np.arange(0, args.tm_relative)          # 预测时间段
    num_regions = len(dats["regions"])                            # 地区数目

    out_trend20_times_relative = np.arange(                       # 迁出趋势时间
        dats["out_trend_t0"]-t0,
        dats["out_trend_t0"]-t0+dats["out_trend20"].shape[0]
    )
    out_trend20_dict = {}
    for i, t in enumerate(out_trend20_times_relative):
        out_trend20_dict[t] = dats["out_trend20"][i, :]

    """ 构建、或读取、或训练模型(逐个省份进行) """
    reg = "山东"
    reg_ind = dats["regions"].index(reg)

    # 准备该地区使用的数据
    protect_t0_relative_reg = protect_t0_relative[region_index]

    model = SEAIRQ(
        populations=dats["population"], y0for1=[0]*8,
        protect=True, protect_args={"t0": },
        De=args.De, Dq=args.Dq, populations=dats["population"],
        y0for1=args.y0, alpha_I=args.alpha_I, alpha_E=args.alpha_E,
        protect=True,
        protect_args={"t0": protect_t0_relative, "k": 0.0},
        score_type=args.fit_score,
        gamma_func_kwargs={"protect_t0": protect_t0_relative,
                            "gammas": 0.06},
        Pmn_func_kwargs={"pmn": pmn_matrix_relative}
    )
    if args.model == "fit":
        # 设置我们拟合模型需要的数据
        if args.use_whhb:
            mask = None
        else:
            mask = np.full(num_regions, True, dtype=np.bool)
            mask[0] = False
        fit_start_index = args.fit_start_relative - epi_t0_relative
        score_kwargs = {
            "times": epi_times_relative[fit_start_index:],
            "true_infects": dats["trueH"][fit_start_index:, :],
            "mask": mask,
        }
        # 搜索
        if args.fit_method == "geatpy":
            fit_kwargs = {
                "method": "geatpy",
                "fig_dir": args.save_dir+"/",
                "njobs": -1,
                "NIND": 4000
            }
        else:
            fit_kwargs = {
                "callback": utils.callback, "method": "annealing"
            }
        dim, lb, ub = model.fit_params_range()
        opt_res = find_best(
            lambda x: score_func(x, model, score_kwargs),
            dim, lb, ub, **fit_kwargs
        )

            # 把拟合得到的参数整理成dataframe，然后保存
            temp_d, temp_i = {}, 0
            for i, (k, vs) in enumerate(model.fit_params_info.items()):
                params_k = opt_res["BestParam"][temp_i:(temp_i+vs[0])]
                for j, v in enumerate(params_k):
                    temp_d[k+str(j)] = v
                temp_i += vs[0]
            pd.Series(temp_d).to_csv(
                os.path.join(args.save_dir, "params.csv")
            )
            # 将得到的最优参数设置到模型中，并保存
            model.set_params(opt_res["BestParam"])
            for i, rr in enumerate(dats["regions"]):
                print("%s: %.4f" % (rr, model.protect_args["k"][i]))
            model.save(os.path.join(args.save_dir, "model.pkl"))
            utils.save(opt_res, os.path.join(args.save_dir, "opt_res.pkl"))

    # 预测结果
    pred_prot = model.predict(pred_times_relative)[-1]
    model.protect = False
    pred_nopr = model.predict(pred_times_relative)[-1]

    """ 计算相关指标以及绘制图像 """
    # 预测R0
    pass

    # 计算每个地区的曲线下面积以及面积差,并保存
    auc = under_area(epi_times_relative, dats["trueH"],
                     pred_times_relative, pred_nopr)
    results = {
        "pred_times": pred_times_relative + t0,
        "pred_prot": pred_prot, "pred_nopr": pred_nopr,
        "true_times": epi_times_relative + t0,
        "trueH": dats["trueH"], "auc": auc,
        "regions": dats["regions"]
    }
    utils.save(results, os.path.join(args.save_dir, "pred.pkl"))

    # 为每个地区绘制曲线图
    plot_all(args.regions, results, os.path.join(args.save_dir, "imgs"),
             t0_ord=t0)


if __name__ == "__main__":
    main()
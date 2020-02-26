import os
from collections import OrderedDict

import numpy as np
import pandas as pd

import utils
from plot import under_area, plot_all
from model import InfectiousBase, score_func, find_best


class NetSEIR(InfectiousBase):
    def __init__(
        self, De, Di, y0for1, populations, gamma_func_kwargs,
        Pmn_func_kwargs, alpha_I, alpha_E, protect=False,
        protect_args=None, score_type="mse"
    ):
        """
        一个基于网络的SEIR传染病模型，其参数有以下的部分：
        args：
            De: 潜伏期，天
            Di: 染病期，天
            y0for1: 武汉或湖北的在t0那一天的初始值。
            populations: 各个地区的人口，因为我们得到的是比例，
                想要得到实际人数需要再乘以总人口数
            gamma_func_kwargs： 平均迁出人口比随时间的变化, 这里需要的是该函数的kwargs
            Pmn_func_kwargs：各个地区间的人口流动矩阵，其实也是一个关于t的函数，
                只是每个时间点返回的是一个矩阵，其第nm个元素表示的是从n地区移动
                到m地区所占n地区总流动人口的比例，这里需要输入的是其参数组成的dict
            protect: boolean，是否加入管控项
            protect_args：管控项的各项参数，如果protect，则此必须给出，以dict的形式,
                没有第一个参数t
            alpha_I, alpha_E：传染期和潜伏期的基本传染率系数（就是没有加管控因素的那个常数）
                ，如果是None，则表示此参数不知道，需要自己估计，这时无法predict，需要先
                fit来通过现有数据来估计。如果给出了指定的数值，则可以直接predict。
            score_type： 计算score的方式，mse或nll（-log likelihood）
        """
        super().__init__(
            De, Di, y0for1, populations, gamma_func_kwargs,
            Pmn_func_kwargs, alpha_I, alpha_E, protect,
            protect_args, score_type
        )
        if self.protect and self.protect_args is None:
            raise ValueError("protect_args must not be None!")

        self.gamma_func = utils.GammaFunc2(**self.gamma_func_kwargs)
        self.Pmn_func = utils.PmnFunc(**self.Pmn_func_kwargs)
        self.num_regions = len(self.populations)
        self.theta = 1 / self.De
        self.beta = 1 / self.Di

        # 在准备数据的时候将湖北或武汉放在第一个上
        I0 = np.zeros(self.num_regions)
        I0[0] = self.y0for1 / self.populations[0]
        self.y0 = np.r_[
            np.ones(self.num_regions), np.zeros(self.num_regions), I0
        ]

    def __call__(self, t, SEI):
        """
        t是时间参数，
        SEI分别是各个地区的S、各个地区的E、各个地区的I组成的一维ndarray向量
        """
        SS = SEI[:self.num_regions]
        EE = SEI[self.num_regions:(2*self.num_regions)]
        II = SEI[(2*self.num_regions):]

        if self.protect:
            alpha_E = self.alpha_E * self.protect_decay(t, **self.protect_args)
            alpha_I = self.alpha_I * self.protect_decay(t, **self.protect_args)
        else:
            alpha_E, alpha_I = self.alpha_E, self.alpha_I

        s2e_i = alpha_I * SS * II
        s2e_e = alpha_E * EE * SS
        e2i = self.theta * EE
        i2r = self.beta * II

        pmnt = self.Pmn_func(t)
        gamma_t = self.gamma_func(t)
        # 这里SS.dot(pmnt)，其运算是将SS看做是1xn维矩阵和pmnt进行的矩阵乘
        s_out_people = gamma_t * (SS.dot(pmnt) - SS)
        e_out_people = gamma_t * (EE.dot(pmnt) - EE)
        i_out_people = gamma_t * (II.dot(pmnt) - II)

        delta_s = - s2e_i - s2e_e + s_out_people
        delta_e = s2e_e + s2e_i + e_out_people - e2i
        delta_i = e2i - i2r + i_out_people
        output = np.r_[delta_s, delta_e, delta_i]
        return output

    def predict(self, times):
        SEI = super().predict(times)
        SS = SEI[:, :self.num_regions] * self.populations
        EE = SEI[:, self.num_regions:(2*self.num_regions)] * self.populations
        II = SEI[:, (2*self.num_regions):] * self.populations
        return SS, EE, II

    @staticmethod
    def protect_decay(t, **kwargs):
        # return utils.protect_decay1(t, t0, tm, eta)
        return utils.protect_decay2(t, **kwargs)

    def score(self, times, true_infects, mask=None):
        """
        因为这里是多个地区的预测，所以true_infects也是一个二维矩阵，即
        shape = num_times x num_regions
        """
        _, _, preds = self.predict(times)
        if mask is not None:
            preds, true_infects = preds[:, mask], true_infects[:, mask]
        if self.score_type == "mse":
            return np.mean((true_infects - preds) ** 2)
        elif self.score_type == "nll":
            return np.mean(preds - np.log(preds) * true_infects)
        else:
            raise ValueError

    def R0(self, ts):
        raise NotImplementedError

    @property
    def fit_params_info(self):
        """
        1. 这里记录我们需要更新的参数的信息，如果想要变换我们更新的参数，就在这里更改，来方便
        程序的实验。
        2. 这里使用OrderDict进行记录，键为其对应的属性名，而值是这个参数的(维度, 下限，上限)
        3. 如果key使用A-B的格式，则这里表示的是self.A["B"]的值
        """
        params = OrderedDict()
        # params["alpha_E"] = (1, 0, 0.1)
        params["alpha_I"] = (1, 0, 1)
        # params["protect_args-eta"] = (31, 0, 1)
        # params["protect_args-tm"] = (31, 0, 31)
        params["protect_args-k"] = (31, 0, 0.5)
        params["y0for1"] = (1, 1, 100)
        return params


def main():

    """ 命令行参数及其整理 """
    parser = utils.MyArguments()
    parser.add_argument("--De", default=5, type=float)
    parser.add_argument("--Di", default=14, type=float)
    parser.add_argument("--alpha_E", default=0.0, type=float)
    parser.add_argument("--alpha_I", default=0.4, type=float)
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

    """ 构建、或读取、或训练模型 """
    # 根据不同的情况来得到合适的模型
    if args.model is not None and args.model != "fit":
        model = NetSEIR.load(args.model)
    else:
        model = NetSEIR(
            De=args.De, Di=args.Di, populations=dats["population"],
            y0for1=args.y0, alpha_I=args.alpha_I, alpha_E=args.alpha_E,
            protect=True,
            protect_args={"t0": protect_t0_relative, "k": 0.0},
            score_type=args.fit_score,
            gamma_func_kwargs={"protect_t0": protect_t0_relative,
                               "gammas": 0.1255},
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
                    "NIND": 400
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

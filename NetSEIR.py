import os
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from scipy.integrate import trapz

import utils
from model import InfectiousBase, find_best
from plot import plot_one_regions

from widgets import GammaFunc, PmnFunc


class NetSEIR(InfectiousBase):
    def __init__(
        self, De, Di, y0, gamma_func_args, Pmn_func_args, alpha_I=None,
        alpha_E=None, protect=False, protect_args=None,
        num_people=None, score_type="mse", fit_method="scipy-NM"
    ):
        """
        一个基于网络的SEIR传染病模型，其参数有以下的部分：
        args：
            De: 潜伏期，天
            Di: 染病期，天
            y0: 初始值。
            gamma_func_args： 平均迁出人口比，是一个关于时间的函数，这里需要输入的是其参数组成的tuple
            Pmn_func：各个地区间的人口流动矩阵，其实也是一个关于t的函数，只是每个时间点返回的是一
                个矩阵，其第nm个元素表示的是从n地区移动到m地区所占n地区总流动人口的比例，这里需要输入的
                是其参数组成的tuple
            protect: boolean，是否加入管控项
            protect_args：管控项的各项参数，如果protect，则此必须给出，以dict的形式,
                没有第一个参数t
            alpha_I, alpha_E：传染期和潜伏期的基本传染率系数（就是没有加管控因素的那个常数）
                ，如果是None，则表示此参数不知道，需要自己估计，这时无法predict，需要先
                fit来通过现有数据来估计。如果给出了指定的数值，则可以直接predict。
            num_people: 各个地区的人口，因为我们得到的是比例，想要得到实际人数需要再乘以
                总人口数
            score_type： 计算score的方式，mse或nll（-log likelihood）
            fit_method： 拟合参数时使用的优化方法
        """
        super().__init__(
            De, Di, y0, gamma_func_args, Pmn_func_args, alpha_I, alpha_E,
            protect, protect_args, num_people, score_type, fit_method
        )
        if self.protect and self.protect_args is None:
            raise ValueError("protect_args must not be None!")

        self.gamma_func = GammaFunc(*self.gamma_func_args)
        self.Pmn_func = PmnFunc(*self.Pmn_func_args)
        self.num_regions = self.Pmn_func(0).shape[0]
        # 实际使用的是theta和beta，所以De和Di是不能作为参数来fit的
        self.theta = 1 / self.De
        self.beta = 1 / self.Di

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
        SS = SEI[:, :self.num_regions]
        EE = SEI[:, self.num_regions:(2*self.num_regions)]
        II = SEI[:, (2*self.num_regions):]
        if self.num_people is not None:
            SS = SS * self.num_people
            EE = EE * self.num_people
            II = II * self.num_people
        return SS, EE, II

    @staticmethod
    def protect_decay2(t, t0, eta, tm, epsilon=0.001):
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
        tmm = tm - t0
        r = 2 * np.log((1-epsilon) / epsilon) / tmm
        x0 = t0 + tmm / 2
        decay = eta / (1 + np.exp(r * (t - x0))) + 1 - eta
        return decay

    @staticmethod
    def protect_decay(t, t0, k):
        le_bool = t <= t0
        return le_bool + (1 - le_bool) * np.exp(-k * (t - t0))

    @staticmethod
    def protect_decay1(t, t0, k):
        pass

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
        FVs = []
        for t in ts:
            pmnt = self.Pmn_func(t)
            gammat = self.gamma_func(t)

            A = np.diag(np.full(self.num_regions, self.theta - gammat))
            A = A - pmnt.T
            A = np.linalg.inv(A)
            B = np.diag(np.full(self.num_regions, self.beta - gammat))
            B = B - pmnt.T
            B = np.linalg.pinv(B)
            AB = np.matmul(A, B)
            FV_1 = self.alpha_E * A + self.alpha_I * self.theta * AB
            FVs.append(FV_1)
        FVs = np.stack(FVs)
        eigs, _ = np.linalg.eig(FVs)
        return abs(eigs).max(axis=1)


def set_model(model, params):
    model.alpha_E = model.kwargs["alpha_E"] = params[0]
    model.alpha_I = model.kwargs["alpha_I"] = params[1]
    model.protect_args["k"] = model.kwargs["protect_args"]["k"] = params[2:]


def score_func(params, model, true_times, true_values, mask=None):
    model_copy = deepcopy(model)
    set_model(model_copy, params)
    return model_copy.score(true_times, true_values, mask=None)

    # def save_opt_res(self, save_file):
    #     utils.save(self.opt_res, save_file, "pkl")


if __name__ == "__main__":

    """ 命令行参数及其整理 """
    parser = ArgumentParser()
    parser.add_argument("--save_dir")
    parser.add_argument("--region_type", default="city",
                        choices=("city", "province"))
    parser.add_argument(
        "--model", default=None,
        help=("默认是None，即不使用训练的模型，而是直接使用命令行赋予的参数"
              "，不然则读取拟合的参数，命令行赋予的参数无效, 如果是fit，则取进行自动的拟合")
    )
    parser.add_argument(
        "--regions", default=None, nargs="+",
        help=("默认是None，则对于省份或城市都使用不同的默认值，不然，则需要键入需要估计"
              "的地区名。如果是all，则将所有的都计算一下看看")
    )
    parser.add_argument("--t0", default="2019-12-31", help="疫情开始的那天")
    parser.add_argument("--y0", default=100, type=float,
                        help="武汉或湖北在t0那天的感染人数")
    parser.add_argument("--tm", default="2020-03-31", help="需要预测到哪天")
    # =0并且不进行训练，则模型认为潜伏期没有传染性
    parser.add_argument("--alpha_E", default=0.0, type=float)
    parser.add_argument("--alpha_I", default=0.15, type=float)
    parser.add_argument("--De", default=3, type=float)
    parser.add_argument("--Di", default=14, type=float)
    parser.add_argument("--protect_t0", default="2020-01-23")
    parser.add_argument("--protect_k", default=0.001, type=float)
    parser.add_argument("--fit_score", default="nll", choices=["nll", "mse"])
    parser.add_argument("--fit_time_start", default="2020-02-01")
    parser.add_argument("--use_pmn_mean", action="store_true")
    args = parser.parse_args()
    # 整理参数
    save_dir = os.path.join("./RESULTS/", args.save_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if args.regions is None:
        if args.region_type == "city":
            plot_regions = [
                "武汉", "孝感", "荆州", "荆门", "随州", "黄石", "宜昌",
                "鄂州", "北京", "上海", "哈尔滨", "淄博"
            ]
        else:
            plot_regions = ["湖北", "北京", "上海", "广东", "湖南",
                            "浙江", "河南", "山东", "黑龙江"]
    else:
        plot_regions = args.regions
    tm_relative = utils.time_str2diff(args.tm, args.t0)
    # 保存args到路径中
    utils.save(args.__dict__, os.path.join(save_dir, "args.json"), "json")

    """ 读取准备好的数据 """
    if args.region_type == "city":
        dat_file = "./DATA/City.pkl"
    else:
        dat_file = "./DATA/Provinces.pkl"
    dats = utils.load(dat_file, "pkl")

    # 时间调整(我们记录的数据都是以ordinal格式记录，但输入模型中需要以相对格式输入,其中t0是0)
    t0 = utils.time_str2ord(args.t0)  # 疫情开始时间
    # protect_t0_relative = dats["response_time"] - t0  # 防控开始时间
    protect_t0_relative = utils.time_str2diff(args.protect_t0, args.t0)
    epi_t0_relative = dats["epidemic_t0"] - t0  # 第一个确诊病例出现的时间
    pmn_matrix_relative = {(k-t0): v for k, v in dats["pmn"].items()}  # pmn的时间
    epi_times_relative = np.arange(  # 确诊病例时间
        epi_t0_relative, epi_t0_relative + dats["epidemic"].shape[0]
    )
    pred_times_relative = np.arange(0, tm_relative)  # 要预测的时间
    out_trend20_times_relative = np.arange(  # 迁出趋势的时间
        dats["out_trend_t0"]-t0,
        dats["out_trend_t0"]-t0+dats["out_trend20"].shape[0]
    )

    # 将迁出趋势也整理成和pmn一样的dict
    out_trend20_dict = {}
    for i, t in enumerate(out_trend20_times_relative):
        out_trend20_dict[t] = dats["out_trend20"][i, :]

    # 得到湖北或武汉其在regions列表中的位置
    if args.region_type == "city":
        hb_wh_index = dats["regions"].index("武汉")
    else:
        hb_wh_index = dats["regions"].index("湖北")

    # 构造y0
    num_regions = len(dats["regions"])
    y0 = np.zeros(num_regions)
    y0[hb_wh_index] = args.y0
    y0 = y0 / dats["population"]
    y0 = np.r_[np.ones(num_regions), np.zeros(num_regions), y0]

    """ 构建、或读取、或训练模型 """
    # 根据不同的情况来得到合适的模型
    if args.model is not None and args.model != "fit":
        model = NetSEIR.load(args.model)
    else:
        model = NetSEIR(
            args.De, args.Di, y0, (protect_t0_relative,),
            (pmn_matrix_relative, args.use_pmn_mean),
            args.alpha_I, args.alpha_E, protect=True,
            num_people=dats["population"],
            protect_args={"t0": protect_t0_relative, "k": args.protect_k},
            score_type=args.fit_score
        )
        if args.model == "fit":
            # 首先找到我们使用的预测数据，这里不使用湖北的数据，不使用前期的数据，而使用
            #   使用后面的数据
            mask = np.full(num_regions, True, dtype=np.bool)
            mask[hb_wh_index] = False
            use_fit_data_start = utils.time_str2ord(args.fit_time_start) - \
                dats["epidemic_t0"]
            use_fit_data_time = epi_times_relative[use_fit_data_start:]
            use_fit_data_epid = dats["epidemic"][use_fit_data_start:]
            # 设置搜索条件
            x0 = [
                2 + num_regions,
                np.zeros(2 + num_regions),
                np.r_[0.5, 0.5, np.full(num_regions, 0.5)]
            ]
            # 搜索
            best_x, opt_res = find_best(
                lambda x: score_func(
                    x, model, use_fit_data_time, use_fit_data_epid, mask
                ), x0, "geatpy", save_dir + "/"
            )
            # 把k整理成dataframe，然后打印一下
            dfk = pd.DataFrame(
                dict(
                    region=["alpha_E", "alpha_I"] + dats["regions"],
                    k=opt_res["BestParam"]
                )
            )
            print(dfk)
            # 将得到的最优参数设置到模型中，并保存
            set_model(model, best_x)
            model.save(os.path.join(save_dir, "model.pkl"))
            utils.save(opt_res, os.path.join(save_dir, "opt_res.pkl"), "pkl")
    # 预测结果
    _, pred_EE_prot, pred_II_prot = model.predict(pred_times_relative)
    model.protect = False
    _, pred_EE_nopr, pred_II_nopr = model.predict(pred_times_relative)

    """ 计算相关指标以及绘制图像 """
    # 预测R0
    R0s = model.R0(pred_times_relative)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(pred_times_relative, R0s)
    fig.savefig(os.path.join(save_dir, "R0.png"))
    plt.close(fig)
    # 首先把相关结果数据都汇集
    results = {
        "pred_times": pred_times_relative + t0,
        "pred_EE": {"no_protect": pred_EE_nopr, "protect": pred_EE_prot},
        "pred_II": {"no_protect": pred_II_nopr, "protect": pred_II_prot},
        "true_times": epi_times_relative + t0,
        "true_epidemic": dats["epidemic"], "use_k": model.protect_args["k"],
        "alpha_I": model.alpha_I, "alpha_E": model.alpha_E,
        "regions": dats["regions"]
    }
    # 计算每个地区的曲线下面积以及面积差
    pred_II_noprotect_only_true = pred_II_nopr[  # true只是pred的一部分
        np.isin(pred_times_relative, epi_times_relative), :]
    true_areas, pred_areas, diff_areas = [], [], []
    for i, region in enumerate(dats["regions"]):
        true_area = trapz(epi_times_relative, dats["epidemic"][:, i])
        pred_area = trapz(epi_times_relative,
                          pred_II_noprotect_only_true[:, i])
        true_areas.append(true_area)
        pred_areas.append(pred_area)
        diff_areas.append(pred_area - true_area)
    results["area"] = {
        "pred": np.array(pred_areas), "true": np.array(true_areas),
        "diff": np.array(diff_areas)
    }
    # 先得到要画的地区的索引
    if plot_regions[0] == "all":
        plot_regions = [(i, reg) for i, reg in enumerate(dats["regions"])]
    else:
        plot_regions = [(dats["regions"].index(reg), reg)
                        for reg in plot_regions]
    # 绘制每个地区的图片，并保存
    if not os.path.exists(os.path.join(save_dir, "part1")):
        os.mkdir(os.path.join(save_dir, "part1"))
    if not os.path.exists(os.path.join(save_dir, "part2")):
        os.mkdir(os.path.join(save_dir, "part2"))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    time_mask = np.isin(pred_times_relative, epi_times_relative)
    for i, reg in plot_regions:
        # 其中一部分
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        print("%d: %s" % (i, reg))
        plot_one_regions(
            axes[0], pred_times_relative, epi_times_relative,
            pred_EE_nopr[:, i], pred_II_nopr[:, i], dats["epidemic"][:, i],
            reg+" no protect", t0_ord=t0
        )
        plot_one_regions(
            axes[1], pred_times_relative, epi_times_relative,
            pred_EE_prot[:, i], pred_II_prot[:, i], dats["epidemic"][:, i],
            reg+" protect", t0_ord=t0
        )
        plot_one_regions(
            axes[2], epi_times_relative, epi_times_relative,
            pred_EE_prot[time_mask, i], pred_II_nopr[time_mask, i],
            dats["epidemic"][:, i], reg+" no protect part", t0_ord=t0,
            use_log=True
        )
        fig.savefig(os.path.join(save_dir, "part1/%s.png" % reg))
        plt.close(fig)
        # 另一部分
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes = axes.flatten()
        print("%d: %s" % (i, reg))
        plot_one_regions(
            axes[0], pred_times_relative, epi_times_relative,
            pred_EE_nopr[:, i], pred_II_nopr[:, i], dats["epidemic"][:, i],
            reg+" no protect", t0_ord=t0
        )
        plot_one_regions(
            axes[1], epi_times_relative, epi_times_relative,
            pred_EE_prot[time_mask, i], pred_II_nopr[time_mask, i],
            dats["epidemic"][:, i], reg+" no protect part", t0_ord=t0,
            use_log=True
        )
        fig.savefig(os.path.join(save_dir, "part2/%s.png" % reg))
        plt.close(fig)


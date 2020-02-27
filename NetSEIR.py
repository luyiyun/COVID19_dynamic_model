import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import utils
from plot import under_area, plot_one_regions
from model import InfectiousBase, score_func, find_best


class NetSEIR(InfectiousBase):
    def __init__(
        self, De, Di, y0for1, populations, gamma_func_kwargs,
        Pmn_func_kwargs, alpha_I, alpha_E, protect=False,
        protect_args=None, score_type="mse"
    ):
        """
        一个基于复杂人口迁徙网络的SEIR模型。

        Arguments:
            De {float} -- 平均潜伏期
            Di {float} -- 平均患病期
            y0for1 {float} -- t0时刻，region-1的患病（I）人数
            populations {ndarray} -- 每个region的人口
            gamma_func_kwargs {dict} -- 用于GammaFunc函数实例化的参数，表示人口迁出
                比的变化趋势
            Pmn_func_kwargs {dict} -- 用于PmnFunc函数实例化的参数，表示各个region到
                其他regions人口迁出率的变化
            alpha_I {float} -- 患病者的感染率系数
            alpha_E {float} -- 潜伏者的感染率系数

        Keyword Arguments:
            protect {bool} -- 是否加入防控措施 (default: {False})
            protect_args {dict} -- 防控措施函数需要的参数，除了时间t (default: {None})
            score_type {str} -- 使用的score类型， mse或nll (default: {"mse"})

        Raises:
            ValueError: [description]
            ValueError: [description]
            NotImplementedError: [description]

        Returns:
            NetSEIR对象 -- 用于接下来的拟合、预测
        """
        super().__init__(
            De, Di, y0for1, populations, gamma_func_kwargs,
            Pmn_func_kwargs, alpha_I, alpha_E, protect,
            protect_args, score_type
        )
        if self.protect and self.protect_args is None:
            raise ValueError("protect_args must not be None!")

        self.gamma_func = utils.GammaFunc1(**self.gamma_func_kwargs)
        self.Pmn_func = utils.PmnFunc(**self.Pmn_func_kwargs)
        self.num_regions = len(self.populations)
        self.theta = 1 / self.De
        self.beta = 1 / self.Di

        # 在准备数据的时候将湖北或武汉放在第一个上
        I0 = np.zeros(self.num_regions)
        I0[0] = self.y0for1
        self.y0 = np.r_[
            self.populations - I0, np.zeros(self.num_regions), I0,
            np.zeros(self.num_regions)
        ]

    def __call__(self, t, SEIR):
        SS = SEIR[:self.num_regions]
        EE = SEIR[self.num_regions:(2*self.num_regions)]
        II = SEIR[(2*self.num_regions):(3*self.num_regions)]
        RR = SEIR[(3*self.num_regions):]
        Nt = SS + EE + II + RR

        if self.protect:
            alpha_E = self.alpha_E * self.protect_decay(t, **self.protect_args)
            alpha_I = self.alpha_I * self.protect_decay(t, **self.protect_args)
        else:
            alpha_E, alpha_I = self.alpha_E, self.alpha_I

        s2e_i = alpha_I * SS * II / Nt
        s2e_e = alpha_E * EE * SS / Nt
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
        delta_r = i2r
        output = np.r_[delta_s, delta_e, delta_i, delta_r]
        return output

    def predict(self, times):
        SEIR = super().predict(times)
        SS = SEIR[:, :self.num_regions]
        EE = SEIR[:, self.num_regions:(2*self.num_regions)]
        II = SEIR[:, (2*self.num_regions):(3*self.num_regions)]
        RR = SEIR[:, (3*self.num_regions):]

        return SS, EE, II, RR

    @staticmethod
    def protect_decay(t, **kwargs):
        """
        防控措施导致传染率系数变化为原来的百分比，随时间变化

        Arguments:
            t {float} -- 时间点

        Returns:
            float -- 在0-1之间
        """
        return utils.protect_decay2(t, **kwargs)

    def score(self, times, true_infects, mask=None):
        preds = self.predict(times)[2]
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
        params = OrderedDict()
        params["alpha_E"] = (1, 0, 0.5)
        params["alpha_I"] = (1, 0, 0.5)
        # params["protect_args-eta"] = (31, 0, 1)
        # params["protect_args-tm"] = (31, 0, 31)
        params["protect_args-k"] = (31, 0, 1)
        params["y0for1"] = (1, 1, 10)
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
    dataset = utils.Dataset(dat_file, args.t0, args.tm, args.fit_time_start)

    """ 构建、或读取、或训练模型 """
    # 根据不同的情况来得到合适的模型
    if args.model is not None and args.model != "fit":
        model = NetSEIR.load(args.model)
    else:
        model = NetSEIR(
            De=args.De, Di=args.Di, populations=dataset.populations,
            y0for1=args.y0, alpha_I=args.alpha_I, alpha_E=args.alpha_E,
            protect=True, score_type=args.fit_score,
            protect_args={
                "t0": dataset.protect_t0.relative,
                "k": 0.0
            },
            gamma_func_kwargs={
                # "protect_t0": dataset.protect_t0.relative,
                # "gammas": 0.06
                "gammas": dataset.out20_dict
            },
            Pmn_func_kwargs={"pmn": dataset.pmn_matrix_relative}
        )
        if args.model == "fit":
            # 设置我们拟合模型需要的数据
            if args.use_whhb:
                mask = None
            else:
                mask = np.full(dataset.num_regions, True, dtype=np.bool)
                mask[0] = False
            fit_start_index = dataset.fit_start_t.ord - dataset.epi_t0.ord
            score_kwargs = {
                "times": dataset.epi_times.relative[fit_start_index:],
                "true_infects": dataset.trueH[fit_start_index:, :],
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
            # for i, rr in enumerate(dats["regions"]):
            #     print("%s: %.4f" % (rr, model.protect_args["k"][i]))
            model.save(os.path.join(args.save_dir, "model.pkl"))
            utils.save(opt_res, os.path.join(args.save_dir, "opt_res.pkl"))

    # 预测结果
    prot_preds = model.predict(dataset.pred_times.relative)
    model.protect = False
    nopr_preds = model.predict(dataset.pred_times.relative)

    """ 计算相关指标以及绘制图像 """
    # 预测R0
    pass

    # 计算每个地区的曲线下面积以及面积差,并保存
    auc = under_area(
        dataset.epi_times.relative,
        dataset.trueH,
        dataset.pred_times.relative,
        nopr_preds[2],
    )
    utils.save(auc, os.path.join(args.save_dir, "auc.pkl"))

    # 为每个地区绘制曲线图
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    img_dir = os.path.join(args.save_dir, "imgs")
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    for i, reg in enumerate(dataset.regions):
        plot_one_regions(
            reg, [
                ("true", dataset.epi_times.ord, dataset.trueH[:, i], "ro"),
                ("predI", dataset.pred_times.ord, prot_preds[2][:, i], "r"),
                ("predE", dataset.pred_times.ord, prot_preds[1][:, i], "y"),
                ("predR", dataset.pred_times.ord, prot_preds[3][:, i], "b")
            ],
            [
                ("true", dataset.epi_times.ord, dataset.trueH[:, i], "ro"),
                ("predI", dataset.pred_times.ord, nopr_preds[2][:, i], "r"),
                ("predE", dataset.pred_times.ord, nopr_preds[1][:, i], "y"),
                ("predR", dataset.pred_times.ord, nopr_preds[3][:, i], "b")
            ],
            save_dir=img_dir
        )


if __name__ == "__main__":
    main()

import os
from copy import deepcopy
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

        # 在一级响应之前，允许I的流动，之后截断其流动
        self.I_flow_func = lambda t: t < self.protect_args["t0"]

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

        pmnt = self.Pmn_func(t) - np.diag(np.ones(self.num_regions))
        gamma_t = self.gamma_func(t)
        # 这里SS.dot(pmnt)，其运算是将SS看做是1xn维矩阵和pmnt进行的矩阵乘
        # s_out_people = gamma_t * (SS.dot(pmnt) - SS)
        # e_out_people = gamma_t * (EE.dot(pmnt) - EE)
        # i_out_people = gamma_t * (II.dot(pmnt) - II)
        s_out_people = (SS * gamma_t).dot(pmnt)
        e_out_people = (EE * gamma_t).dot(pmnt)
        i_out_people = (II * gamma_t).dot(pmnt) * self.I_flow_func(t)

        delta_s = - s2e_i - s2e_e + s_out_people
        delta_e = s2e_e + s2e_i + e_out_people - e2i
        delta_i = e2i - i2r + i_out_people
        delta_r = i2r
        output = np.r_[delta_s, delta_e, delta_i, delta_r]
        return output

    def predict(self, times):
        SEIR = super().predict(np.array(times))
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
            return np.mean(preds - np.log(preds+1e-4) * true_infects)
        elif self.score_type == "mae":
            return np.mean(np.abs(true_infects - preds))
        else:
            raise ValueError

    def R0(self, ts=None, protect=False, relative=False):
        """
        仅计算的是人口流动对R0的影响
        relative=True则计算的时候会考虑到当前易感人群的比例
        """
        if ts is None:
            ts = list(self.gamma_func.gammas.keys())

        if relative:
            ss, ee, ii, rr = self.predict(np.array(ts))
            nn = ss + ee + ii + rr
            sn = ss / nn

        R0s = []
        for i, t in enumerate(ts):
            pmnt = self.Pmn_func(t)
            gammat = self.gamma_func(t)

            p = pmnt - np.diag(np.ones(self.num_regions))
            p = np.matmul(p.T, np.diag(gammat))

            A = np.diag(np.full((self.num_regions,), self.theta)) - p
            A_1 = np.linalg.inv(A)
            D = np.diag(np.full((self.num_regions,), self.beta)) - p
            D_1 = np.linalg.inv(D)
            C_1 = np.matmul(D_1, A_1) * self.theta

            if protect:
                protect_mat = np.diag(
                    self.protect_decay(t, **self.protect_args))
                A_1 = np.matmul(protect_mat, A_1)
                C_1 = np.matmul(protect_mat, C_1)
            if relative:
                sn_mat = np.diag(sn[i])
                A_1 = np.matmul(sn_mat, A_1)
                C_1 = np.matmul(sn_mat, C_1)

            G = self.alpha_E * A_1 + self.alpha_I * C_1
            eigs, _ = np.linalg.eig(G)
            R0s.append(np.abs(eigs).max())
        return R0s

    @property
    def fit_params_info(self):
        params = OrderedDict()
        # params["alpha_E"] = (1, 0, 0.5)
        params["alpha_I"] = (31, 0, 0.8)
        # params["protect_args-eta"] = (31, 0, 1)
        # params["protect_args-tm"] = (31, 0, 31)
        # params["protect_args-k"] = (31, 0, 10)
        # params["y0for1"] = (1, 1, 5)
        return params


def main():

    """ 命令行参数及其整理 """
    parser = utils.MyArguments()
    parser.add_argument("--De", default=5.2, type=float)
    parser.add_argument("--Di", default=11.5, type=float)
    parser.add_argument("--alpha_E", default=0.0, type=float)
    parser.add_argument("--alpha_I", default=0.4, type=float)
    parser.add_argument("--protect_k", default=0.0, type=float)
    parser.add_argument("--use_19", action="store_true")
    parser.add_argument("--zero_spring", action="store_true")
    args = parser.parse_args()  # 对于一些通用的参数，这里已经进行整理了

    """ 读取准备好的数据 """
    # if args.region_type == "city":
    #     dat_file = "./DATA/City.pkl"
    # else:
    dat_file = "./DATA/Provinces.pkl"
    dataset = utils.Dataset(dat_file, args.t0, args.tm, args.fit_time_start)

    """ 构建、或读取、或训练模型 """
    # 根据不同的情况来得到合适的模型
    if args.model is not None:
        model = NetSEIR.load(args.model)
    else:
        model = NetSEIR(
            De=args.De, Di=args.Di, populations=dataset.populations,
            y0for1=args.y0, alpha_I=args.alpha_I, alpha_E=args.alpha_E,
            protect=True, score_type=args.fit_score,
            protect_args={
                "t0": dataset.protect_t0.delta, "k": args.protect_k
            },
            gamma_func_kwargs={
                "gammas": (dataset.out19_dict if args.use_19
                           else dataset.out20_dict),
                "zero_period": (dataset.zero_period.delta
                                if args.zero_spring else None)
            },
            Pmn_func_kwargs={"pmn": dataset.pmn_matrix_relative}
        )
    if args.fit:
        # 设置我们拟合模型需要的数据
        if args.use_whhb:
            mask = None
        else:
            mask = np.full(dataset.num_regions, True, dtype=np.bool)
            mask[0] = False
        fit_start_index = (dataset.fit_start_t.ord - dataset.epi_t0.ord)
        fit_start_index = int(fit_start_index)
        score_kwargs = {
            "times": dataset.epi_times.delta[fit_start_index:],
            "true_infects": dataset.trueH[fit_start_index:, :],
            "mask": mask,
        }
        # 搜索
        if args.fit_method == "annealing":
            fit_kwargs = {
                "callback": utils.callback, "method": "annealing"
            }
        else:
            fit_kwargs = {
                "method": "SEGA",
                "fig_dir": args.save_dir+"/",
                "njobs": -1,
                "NIND": args.geatpy_nind,
                "MAXGEN": args.geatpy_maxgen,
                "n_populations": args.geatpy_npop
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
        model.save(os.path.join(args.save_dir, "model.pkl"))
        utils.save(opt_res, os.path.join(args.save_dir, "opt_res.pkl"))

    # 预测结果
    prot_preds = model.predict(dataset.pred_times.delta)
    model.protect = False
    nopr_preds = model.predict(dataset.pred_times.delta)

    """ 计算相关指标以及绘制图像 """
    # 预测R0
    pass

    # 计算每个地区的曲线下面积以及面积差,并保存
    auc = under_area(
        dataset.epi_times.delta, dataset.trueH,
        dataset.pred_times.delta, nopr_preds[2],
    )
    auc_df = pd.DataFrame(
        auc.T, columns=["true_area", "pred_area", "diff_area"],
        index=dataset.regions
    )
    auc_df["population"] = dataset.populations
    auc_df["diff_norm"] = auc_df.diff_area / auc_df.population
    auc_df.sort_values("diff_norm", inplace=True)
    # utils.save(auc, os.path.join(args.save_dir, "auc.pkl"))

    # 为每个地区绘制曲线图
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    img_dir = os.path.join(args.save_dir, "imgs")
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    for i, reg in enumerate(dataset.regions):
        plot_one_regions(
            reg, [
                ("true", dataset.epi_times.ord.astype("int"),
                 dataset.trueH[:, i], "ro"),
                ("predI", dataset.pred_times.ord.astype("int"),
                 prot_preds[2][:, i], "r"),
                ("predE", dataset.pred_times.ord.astype("int"),
                 prot_preds[1][:, i], "y"),
                ("predR", dataset.pred_times.ord.astype("int"),
                 prot_preds[3][:, i], "b")
            ],
            [
                ("true", dataset.epi_times.ord.astype("int"),
                 dataset.trueH[:, i], "ro"),
                ("predI", dataset.pred_times.ord.astype("int"),
                 nopr_preds[2][:, i], "r"),
                ("predE", dataset.pred_times.ord.astype("int"),
                 nopr_preds[1][:, i], "y"),
                ("predR", dataset.pred_times.ord.astype("int"),
                 nopr_preds[3][:, i], "b")
            ],
            save_dir=img_dir
        )

    # 保存结果
    for i, name in enumerate(["predS", "predE", "predI", "predR"]):
        pd.DataFrame(
            prot_preds[i],
            columns=dataset.regions,
            index=dataset.pred_times.str
        ).to_csv(
            os.path.join(args.save_dir, "protect_%s.csv" % name)
        )
        pd.DataFrame(
            nopr_preds[i],
            columns=dataset.regions,
            index=dataset.pred_times.str
        ).to_csv(
            os.path.join(args.save_dir, "noprotect_%s.csv" % name)
        )
    auc_df.to_csv(os.path.join(args.save_dir, "auc.csv"))
    # 这里保存的是原始数据
    for i, attr_name in enumerate(["trueD", "trueH", "trueR"]):
        save_arr = getattr(dataset, attr_name)
        pd.DataFrame(
            save_arr,
            columns=dataset.regions,
            index=dataset.epi_times.str
        ).to_csv(os.path.join(args.save_dir, "%s.csv" % attr_name))
    # 保存args到路径中（所有事情都完成再保存数据，安全）
    save_args = deepcopy(args.__dict__)
    save_args["model_type"] = "NetSEIR"
    utils.save(save_args, os.path.join(args.save_dir, "args.json"), "json")


if __name__ == "__main__":
    main()

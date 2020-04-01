import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import InfectiousBase, find_best, score_func
import utils
from plot import under_area, plot_one_regions


class NetSEAIRQ(InfectiousBase):
    def __init__(
        self, populations, y0for1, protect, protect_args,
        gamma_func_kwargs, Pmn_func_kwargs,
        De=5, Dq=14, c=13.0046, beta=2.03e-9, q=1.88e-7,
        rho=0.6834, deltaI=0.1328, deltaQ=0.1259, gammaI=0.1029, gammaA=0.2978,
        gammaH=0.1024, alpha=0.0009, theta=1.6003, nu=1.5008, score_type="mse"
    ):
        """
        y0 = [H R D E A I Sq Eq] + [S]
        """
        super().__init__(
            populations, y0for1, protect, protect_args, gamma_func_kwargs,
            Pmn_func_kwargs, De, Dq, c, beta, q, rho, deltaI, deltaQ, gammaI,
            gammaA, gammaH, alpha, theta, nu, score_type
        )
        self.sigma = 1 / De
        self.lam = 1 / Dq
        self.GammaFunc = utils.GammaFunc1(**self.gamma_func_kwargs)
        self.PmnFunc = utils.PmnFunc(**self.Pmn_func_kwargs)

        # 计算y0
        self.num_regions = len(populations)
        y0s = []
        y0s_remain = populations
        for i in y0for1:
            y0s_i = np.r_[i, np.zeros(self.num_regions-1)]
            y0s_remain = y0s_remain - y0s_i
            y0s.append(y0s_i)
        y0s.append(y0s_remain)
        self.y0 = np.concatenate(y0s)

    def __call__(self, t, ALL):
        """
        t是时间参数，
        y0 = [H R D E A I Sq Eq] + [S]
        """
        HH = ALL[:self.num_regions]
        RR = ALL[self.num_regions:2*self.num_regions]
        DD = ALL[2*self.num_regions:3*self.num_regions]
        EE = ALL[3*self.num_regions:4*self.num_regions]
        AA = ALL[4*self.num_regions:5*self.num_regions]
        II = ALL[5*self.num_regions:6*self.num_regions]
        SSq = ALL[6*self.num_regions:7*self.num_regions]
        EEq = ALL[7*self.num_regions:8*self.num_regions]
        SS = ALL[8*self.num_regions:]
        Nt = HH + RR + DD + EE + AA + II + SSq + EEq + SS

        # 如果有保护措施，这里计算受到保护措施影响的c和q
        if self.protect:
            decayc, decayq = self.protect_decay(t, **self.protect_args)
            ci, qi = self.c * decayc, self.q * decayq
        else:
            ci, qi = self.c, self.q

        # 计算IAES的移动人口补充数量
        PmnT = self.PmnFunc(t)
        GammaT = self.GammaFunc(t)
        SS_out = GammaT * (SS.dot(PmnT) - SS)
        II_out = GammaT * (II.dot(PmnT) - II)
        EE_out = GammaT * (EE.dot(PmnT) - EE)
        AA_out = GammaT * (AA.dot(PmnT) - AA)

        # 计算导数
        SIAE = SS * (II + self.theta * AA + self.nu * EE) / Nt
        HH_ = self.deltaI*II+self.deltaQ*EEq-(self.alpha+self.gammaH)*HH
        RR_ = self.gammaI*II+self.gammaA*AA+self.gammaH*HH
        DD_ = self.alpha * (II + HH)
        EE_ = self.beta*ci*(1-qi)*SIAE-self.sigma*EE+EE_out
        AA_ = self.sigma*(1-self.rho)*EE-self.gammaA*AA+AA_out
        II_ = self.sigma*self.rho*EE-(self.deltaI+self.alpha+self.gammaI)*II +\
            II_out
        SSq_ = (1-self.beta)*ci*qi*SIAE-self.lam*SSq
        EEq_ = self.beta*ci*qi*SIAE-self.deltaQ*EEq
        SS_ = -(self.beta*ci+ci*qi*(1-self.beta))*SIAE+self.lam*SSq+SS_out

        output = np.r_[HH_, RR_, DD_, EE_, AA_, II_, SSq_, EEq_, SS_]
        return output

    def predict(self, times):
        ALL = super().predict(times)
        HH = ALL[:, :self.num_regions]
        RR = ALL[:, self.num_regions:2*self.num_regions]
        DD = ALL[:, 2*self.num_regions:3*self.num_regions]
        EE = ALL[:, 3*self.num_regions:4*self.num_regions]
        AA = ALL[:, 4*self.num_regions:5*self.num_regions]
        II = ALL[:, 5*self.num_regions:6*self.num_regions]
        SSq = ALL[:, 6*self.num_regions:7*self.num_regions]
        EEq = ALL[:, 7*self.num_regions:8*self.num_regions]
        SS = ALL[:, 8*self.num_regions:]
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
        raise NotImplementedError

    @property
    def fit_params_info(self):
        params = OrderedDict()
        params["y0for1[3:6]"] = (3, 0, 100)
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
        params["protect_args-c_k"] = (31, 0, 1)
        params["protect_args-q_k"] = (31, 0, 1)
        return params


def main():

    """ 命令行参数及其整理 """
    parser = utils.MyArguments()
    parser.add_argument("--De", default=5, type=float)
    parser.add_argument("--Dq", default=14, type=float)
    args = parser.parse_args()  # 对于一些通用的参数，这里已经进行整理了

    """ 读取准备好的数据 """
    if args.region_type == "city":
        dat_file = "./DATA/City.pkl"
    else:
        dat_file = "./DATA/Provinces.pkl"
    dataset = utils.Dataset(dat_file, args.t0, args.tm, args.fit_time_start)

    """ 构建、或读取、或训练模型 """
    # 根据不同的情况来得到合适的模型
    if args.model is not None and args.model != "fit":
        model = NetSEAIRQ.load(args.model)
    else:
        model = NetSEAIRQ(
            populations=dataset.populations,
            y0for1=np.array([args.y0, 0, 0, 0, 0, 0, 0, 0]),
            protect=True,
            protect_args={
                "t0": dataset.protect_t0.relative,
                "c_k": 0.0, "q_k": 0.0
            },
            gamma_func_kwargs={"gammas": dataset.out20_dict},
            Pmn_func_kwargs={"pmn": dataset.pmn_matrix_relative},
            De=args.De, Dq=args.Dq, score_type=args.fit_score
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
                "trueH": dataset.trueH[fit_start_index:, :],
                "trueD": dataset.trueD[fit_start_index:, :],
                "trueR": dataset.trueR[fit_start_index:, :],
                "mask": mask,
            }
            # 搜索
            if args.fit_method == "geatpy":
                fit_kwargs = {
                    "method": "geatpy",
                    "fig_dir": args.save_dir+"/",
                    "njobs": -1,
                    "NIND": 400,
                    # "MAXGEN": 50
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
        nopr_preds[0],
    )
    utils.save(auc, os.path.join(args.save_dir, "auc.pkl"))

    # 为每个地区绘制曲线图
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    img_dir = os.path.join(args.save_dir, "imgs")
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    for i, reg in enumerate(dataset.regions):
        """
        y0 = [H R D E A I Sq Eq] + [S]
        """
        plot_one_regions(
            reg, [
                ("true", dataset.epi_times.ord, dataset.trueH[:, i], "ro"),
                ("predH", dataset.pred_times.ord, prot_preds[0][:, i], "r"),
                ("predR", dataset.pred_times.ord, prot_preds[1][:, i], "b"),
                ("predD", dataset.pred_times.ord, prot_preds[2][:, i], "k"),
                ("predE", dataset.pred_times.ord, prot_preds[3][:, i], "y"),
                ("predA", dataset.pred_times.ord, prot_preds[4][:, i], "g"),
                ("predI", dataset.pred_times.ord, prot_preds[4][:, i], "c"),
            ],
            [
                ("true", dataset.epi_times.ord, dataset.trueH[:, i], "ro"),
                ("predH", dataset.pred_times.ord, nopr_preds[0][:, i], "r"),
                ("predR", dataset.pred_times.ord, nopr_preds[1][:, i], "b"),
                ("predD", dataset.pred_times.ord, nopr_preds[2][:, i], "k"),
                ("predE", dataset.pred_times.ord, nopr_preds[3][:, i], "y"),
                ("predA", dataset.pred_times.ord, nopr_preds[4][:, i], "g"),
                ("predI", dataset.pred_times.ord, nopr_preds[4][:, i], "c"),
            ],
            save_dir=img_dir
        )

    # 保存args到路径中（所有事情都完成再保存数据，安全）
    utils.save(args.__dict__, os.path.join(args.save_dir, "args.json"), "json")


if __name__ == "__main__":
    main()

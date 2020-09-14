import os
from math import ceil
from copy import deepcopy
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import utils
from plot import under_area, plot_one_regions
from model import InfectiousBase, find_best, score_func


class NetSEAIRQ_piecewise(InfectiousBase):
    def __init__(
        self, populations, y0_hb, gamma_func_kwargs, Pmn_func_kwargs,
        De=5, Dq=14, c=13.0046, beta=2.03e-9, q=0,
        rho=0.6834, deltaI=0.1328, deltaQ=0.1259, gammaI=0.1029, gammaA=0.2978,
        gammaH=0.1024, theta=1.6003, nu=1.5008, phi=0.9,
        score_type="mse"
    ):
        """
        y0 = [H R E A I Sq Eq] + [S]
        """
        super().__init__(
            populations, y0_hb, gamma_func_kwargs,
            Pmn_func_kwargs, De, Dq, c, beta, q, rho, deltaI, deltaQ, gammaI,
            gammaA, gammaH, theta, nu, phi, score_type
        )
        self.sigma = 1 / De
        self.lam = 1 / Dq
        self.GammaFunc = utils.GammaFunc1(**self.gamma_func_kwargs)
        self.PmnFunc = utils.PmnFunc(**self.Pmn_func_kwargs)

        # 计算y0
        self.num_regions = len(populations)
        y0s = []
        y0s_remain = populations
        for i in y0_hb:
            y0s_i = np.r_[i, np.zeros(self.num_regions-1)]
            y0s_remain = y0s_remain - y0s_i
            y0s.append(y0s_i)
        y0s.append(y0s_remain)
        self.y0 = np.concatenate(y0s)

        # 在一级响应之前，允许I的流动，之后截断其流动
        # self.I_flow_func = lambda t: t < self.protect_args["t0"]

    def __call__(self, t, ALL):
        """
        t是时间参数，
        y0 = [H R E A I Sq Eq] + [S]
        """
        HH = ALL[:self.num_regions]
        RR = ALL[self.num_regions:2*self.num_regions]
        EE = ALL[2*self.num_regions:3*self.num_regions]
        AA = ALL[3*self.num_regions:4*self.num_regions]
        II = ALL[4*self.num_regions:5*self.num_regions]
        SSq = ALL[5*self.num_regions:6*self.num_regions]
        EEq = ALL[6*self.num_regions:7*self.num_regions]
        SS = ALL[7*self.num_regions:]
        Nt = HH + RR + EE + AA + II + SSq + EEq + SS

        ci, qi = self.c, 0  # 完全没有控制的情况下，隔离率是0

        # 计算IAES的移动人口补充数量
        PmnT = self.PmnFunc(t) - np.diag(np.ones(self.num_regions))
        GammaT = self.GammaFunc(t)
        SS_out = (SS * GammaT).dot(PmnT)
        II_out = (II * GammaT).dot(PmnT)
        EE_out = (EE * GammaT).dot(PmnT)
        AA_out = (AA * GammaT).dot(PmnT)  # * self.I_flow_func(t)

        # 计算导数
        SIAE = SS * (II + self.theta * AA + self.nu * EE) / Nt
        HH_ = self.phi*self.deltaI*II + self.deltaQ*EEq - self.gammaH*HH
        RR_ = (1-self.phi)*self.gammaI*II+self.gammaA*AA+self.gammaH*HH
        EE_ = self.beta*ci*(1-qi)*SIAE-self.sigma*EE+EE_out
        AA_ = self.sigma*(1-self.rho)*EE-self.gammaA*AA+AA_out
        II_ = self.sigma*self.rho*EE -\
            (self.phi*self.deltaI+(1-self.phi)*self.gammaI)*II +\
            II_out
        SSq_ = (1-self.beta)*ci*qi*SIAE-self.lam*SSq
        EEq_ = self.beta*ci*qi*SIAE-self.deltaQ*EEq
        SS_ = -(self.beta*ci+ci*qi*(1-self.beta))*SIAE+self.lam*SSq+SS_out

        output = np.r_[HH_, RR_, EE_, AA_, II_, SSq_, EEq_, SS_]
        return output

    def predict(self, times):
        ALL = super().predict(times)
        HH = ALL[:, :self.num_regions]
        RR = ALL[:, self.num_regions:2*self.num_regions]
        EE = ALL[:, 2*self.num_regions:3*self.num_regions]
        AA = ALL[:, 3*self.num_regions:4*self.num_regions]
        II = ALL[:, 4*self.num_regions:5*self.num_regions]
        SSq = ALL[:, 5*self.num_regions:6*self.num_regions]
        EEq = ALL[:, 6*self.num_regions:7*self.num_regions]
        SS = ALL[:, 7*self.num_regions:]
        return HH, RR, EE, AA, II, SSq, EEq, SS

    # @staticmethod
    # def protect_decay(t, t0, c_k, q_k):
    #     decayc = utils.protect_decay2(t, t0, c_k)
    #     decayq = 1 - utils.protect_decay2(t, t0, q_k)  # 隔离率是慢慢从0升上去的
    #     return decayc, decayq

    def score(
        self, times, trueH=None, trueR=None, mask=None
    ):
        true_dat = [trueH, trueR]
        if all([d is None for d in true_dat]):
            raise ValueError("True data can't be all None.")
        pred_dat = self.predict(times)[:2]
        diff = 0
        for t_d, p_d in zip(true_dat, pred_dat):
            if t_d is not None:
                if self.score_type == "mse":
                    diff_i = (t_d - p_d) ** 2
                elif self.score_type == "nll":
                    diff_i = (p_d - np.log(p_d) * t_d)
                elif self.score_type == "mae":
                    diff_i = np.abs(t_d - p_d)
                else:
                    raise ValueError
                diff += diff_i
        if mask is not None:
            diff = diff * mask
        return np.mean(diff)

    def R0(self, ts):
        raise NotImplementedError

    @property
    def fit_params_info(self):
        """
                                                     (      密切接触者     )
        y0 = [H     R     E     A         I          Sq       Eq      ] + [S]
             (住院者 恢复者 潜伏者 无症状感染者 有症状感染者 隔离易感者 隔离潜伏者 易感者)
        beta             = 接触后的传染概率
        q                = 隔离率                   初始隔离率应该是0吧
        c                = 接触率
        rho              = 有症状感染者比例
        deltaI           = 有症状感染者收治的速率
        deltaQ           = 隔离的潜伏者发病(收治)的速率
        gammaI           = 有症状感染者自愈的速度
        gammaA           = 无症状感染者自愈的速度
        gammaH           = 在医院感染者自愈的速度
        phi              = 有症状感染者的收治率
        theta            = 无症状感染者传染率系数
        nu               = 潜伏者传染率系数
        protect_args-c_k = 接触率管控参数
        protect_args-q_k = 隔离率管控措施
        """
        params = OrderedDict()
        params["y0_hb[2:4]"] = (2, 0, 10)
        params["c"] = (31, 0, 100)
        params["beta"] = (1, 0, 1)
        params["q"] = (31, 0, 1)
        params["rho"] = (1, 0, 1)
        params["deltaI"] = (1, 0, 1)
        params["deltaQ"] = (1, 0, 1)
        params["gammaI"] = (1, 0, 1)
        params["gammaA"] = (1, 0, 1)
        params["gammaH"] = (1, 0, 1)
        params["phi"] = (1, 0, 1)
        params["theta"] = (1, 0, 10)
        params["nu"] = (1, 0, 10)
        for n in self._no_fit_name:
            del params[n]
        return params

    def no_fit_params(self, names):
        self._no_fit_name = names

    def set_y0(self, y0):
        self.y0 = y0
        # self.kwargs["y0"] = y0


def main():

    """ 命令行参数及其整理 """
    parser = utils.MyArguments()
    parser.add_argument("--De", default=5.2, type=float, help="平均潜伏期")
    parser.add_argument("--Dq", default=14, type=float, help="平均隔离期")
    parser.add_argument("--c", default=13.0046, type=float, help="初始平均接触率")
    parser.add_argument("--q", default=0.0, type=float, help="初始隔离率")
    parser.add_argument(
        "--beta", default=2.03e-9, type=float, help="基础传染概率"
    )
    parser.add_argument(
        "--theta", default=1.6003, type=float, help="无症状感染者传染概率系数"
    )
    parser.add_argument(
        "--nu", default=1.5008, type=float, help="潜伏期传染概率系数"
    )
    parser.add_argument(
        "--phi", default=0.9, type=float, help="有症状感染者收治率"
    )
    parser.add_argument(
        "--gammaI", default=0.1029, type=float, help="有症状感染者自愈速率"
    )
    parser.add_argument(
        "--gammaA", default=0.2978, type=float, help="无症状感染者自愈速度"
    )
    parser.add_argument(
        "--gammaH", default=1/10.5, type=float, help="医院治愈速率"
    )
    parser.add_argument(
        "--deltaI", default=1/3.5, type=float, help="出现症状患者被收治的速率"
    )
    parser.add_argument(
        "--deltaQ", default=0.1259, type=float,
        help="隔离的潜伏者出现症状（及时被收治）的速率"
    )
    parser.add_argument(
        "--rho", default=0.6834, type=float, help="有症状感染者占所有感染者的比例"
    )
    parser.add_argument("--use_19", action="store_true")
    parser.add_argument("--zero_spring", action="store_true")
    parser.add_argument(
        "-pil", "--piecewise_interval_length", default=3, type=int)
    args = parser.parse_args()  # 对于一些通用的参数，这里已经进行整理了

    """ 读取准备好的数据 """
    dat_file = "./DATA/Provinces.pkl"
    dataset = utils.Dataset(dat_file, args.t0, args.tm, args.fit_time_start)

    """ 构建、或读取、或训练模型 """
    # 根据不同的情况来得到合适的模型
    if args.model is not None and args.model != "fit":
        models = NetSEAIRQ_piecewise.load(args.model)
    else:   # 不然就进行训练
        # 设置我们拟合模型需要的数据
        if args.use_whhb:
            mask = None
        else:
            mask = np.full(dataset.num_regions, True, dtype=np.bool)
            mask[0] = False
        fit_start_index = (dataset.fit_start_t.ord - dataset.epi_t0.ord)
        fit_start_index = int(fit_start_index)
        fit_data_all = dataset.epi_times.delta[fit_start_index:]
        # 根据分段的宽度，设置多个模型，并将其训练用参数也
        pil = args.piecewise_interval_length
        n_models = int(ceil(fit_data_all.shape[0] / pil))
        models = []
        score_kwargs = []
        for i in range(n_models):
            model = NetSEAIRQ_piecewise(
                populations=dataset.populations,
                y0_hb=np.array([0, 0, 0, 0, args.y0, 0, 0]),
                score_type=args.fit_score,
                gamma_func_kwargs={
                    "gammas": (dataset.out19_dict if args.use_19
                               else dataset.out20_dict),
                    "zero_period": (dataset.zero_period.delta
                                    if args.zero_spring else None)
                },
                Pmn_func_kwargs={"pmn": dataset.pmn_matrix_relative},
                De=args.De, Dq=args.Dq, c=args.c, q=args.q, beta=args.beta,
                rho=args.rho, deltaI=args.deltaI, deltaQ=args.deltaQ,
                gammaI=args.gammaI, gammaA=args.gammaH, gammaH=args.gammaH,
                theta=args.theta, nu=args.nu, phi=args.phi,
            )
            use_dat_start = i * pil
            use_dat_end = min((i + 1) * pil, fit_data_all.shape[0])
            score_kwarg = {
                "times": dataset.epi_times.delta[use_dat_start:use_dat_end],
                "mask": mask,
                "trueH": dataset.trueH[use_dat_start:use_dat_end],
                # "trueR": (dataset.trueD + dataset.trueR)[
                #     use_dat_start:use_dat_end]
            }
            models.append(model)
            score_kwargs.append(score_kwarg)
        # 搜索最优参数
        if args.fit_method == "annealing":
            fit_kwargs = {
                "callback": utils.callback, "method": "annealing"
            }
        else:
            fit_kwargs = {
                "method": args.fit_method,
                "fig_dir": args.save_dir+"/",
                "njobs": -1,
                "NIND": args.geatpy_nind,
                "MAXGEN": args.geatpy_maxgen,
                "n_populations": args.geatpy_npop
            }
        last_y0 = None
        predHs = []
        all_opts = []
        for i, (model, score_kwarg) in enumerate(zip(models, score_kwargs)):
            # 被这次训练的时间整理出来
            start_time = utils.CustomDate.from_delta(
                score_kwarg["times"].min(), dataset.t0.str
            )
            end_time = utils.CustomDate.from_delta(
                score_kwarg["times"].max(), dataset.t0.str
            )
            print("开始训练 %s<->%s" % (start_time.str, end_time.str))
            # 第一次训练的模型和后面训练的模型使用不同的参数
            # 之后训练的模型要使用前面模型的最后一天输出作为y0
            if i == 0:
                model.no_fit_params([])
            else:
                model.no_fit_params(["y0_hb[2:4]"])
                model.set_y0(last_y0)
            # 得到训练参数，进行训练
            dim, lb, ub = model.fit_params_range()
            opt_res = find_best(
                lambda x: score_func(x, model, score_kwarg),
                dim, lb, ub, **fit_kwargs
            )
            all_opts.append(opt_res)
            # 将得到的最优参数设置到模型中
            model.set_params(opt_res["BestParam"])
            # 预测结果
            preds = model.predict(score_kwarg["times"])
            predHs.append(preds[0])
            # 预测结果中最后一天作为新的y0
            last_y0 = np.concatenate(preds, axis=1)[-1, :]
    predHs = np.concatenate(predHs, axis=0)
    utils.save(all_opts, os.path.join(args.save_dir, "opt_res.pkl"))
    utils.save(
        [m.kwargs for m in models],
        os.path.join(args.save_dir, "models.pkl")
    )
    # model.save(os.path.join(args.save_dir, "model.pkl"))
    # utils.save(opt_res, os.path.join(args.save_dir, "opt_res.pkl"))
    """ 计算相关指标以及绘制图像 """
    # 预测R0
    pass

    # 计算每个地区的曲线下面积以及面积差,并保存
    # auc = under_area(
    #     dataset.epi_times.delta,
    #     dataset.trueH,
    #     dataset.pred_times.delta,
    #     nopr_preds[0],
    # )
    # auc_df = pd.DataFrame(
    #     auc.T, columns=["true_area", "pred_area", "diff_area"],
    #     index=dataset.regions
    # )
    # auc_df["population"] = dataset.populations
    # auc_df["diff_norm"] = auc_df.diff_area / auc_df.population
    # auc_df.sort_values("diff_norm", inplace=True)

    # 为每个地区绘制曲线图
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    img_dir = os.path.join(args.save_dir, "imgs")
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    for i, reg in enumerate(dataset.regions):
        """
        y0 = [H R E A I Sq Eq] + [S]
        """
        plot_one_regions(
            reg, [
                ("trueH", dataset.epi_times.ord.astype("int"),
                 dataset.trueH[:, i], "ro"),
                # ("trueR", dataset.epi_times.ord.astype("int"),
                #  dataset.trueR[:, i]+dataset.trueD[:, i], "bo"),
                # ("predH", dataset.pred_times.ord.astype("int"),
                #  predHs[:, i], "r"),
                ("predH", dataset.epi_times.ord.astype("int"),
                 predHs[:, i], "r"),
                # ("predR", dataset.pred_times.ord.astype("int"),
                #  prot_preds[1][:, i], "b"),
                # ("predE", dataset.pred_times.ord.astype("int"),
                #  prot_preds[3][:, i], "y"),
                # ("predA", dataset.pred_times.ord.astype("int"),
                #  prot_preds[4][:, i], "g"),
                # ("predI", dataset.pred_times.ord.astype("int"),
                #  prot_preds[4][:, i], "c"),
            ],
            [
                ("trueH", dataset.epi_times.ord.astype("int"),
                 dataset.trueH[:, i], "ro"),
                # ("trueR", dataset.epi_times.ord.astype("int"),
                #  dataset.trueR[:, i]+dataset.trueD[:, i], "bo"),
                # ("predH", dataset.pred_times.ord.astype("int"),
                #  predHs[:, i], "r"),
                ("predH", dataset.epi_times.ord.astype("int"),
                 predHs[:, i], "r"),
                # ("predR", dataset.pred_times.ord.astype("int"),
                #  nopr_preds[1][:, i], "b"),
                # ("predE", dataset.pred_times.ord.astype("int"),
                #  nopr_preds[3][:, i], "y"),
                # ("predA", dataset.pred_times.ord.astype("int"),
                #  nopr_preds[4][:, i], "g"),
                # ("predI", dataset.pred_times.ord.astype("int"),
                #  nopr_preds[4][:, i], "c"),
            ],
            save_dir=img_dir
        )
    # # 保存结果
    # for i, name in enumerate([
    #     "predH", "predR", "predE", "predA", "predI"
    # ]):
    #     pd.DataFrame(
    #         prot_preds[i],
    #         columns=dataset.regions,
    #         index=dataset.pred_times.str
    #     ).to_csv(
    #         os.path.join(args.save_dir, "protect_%s.csv" % name)
    #     )
    #     pd.DataFrame(
    #         nopr_preds[i],
    #         columns=dataset.regions,
    #         index=dataset.pred_times.str
    #     ).to_csv(
    #         os.path.join(args.save_dir, "noprotect_%s.csv" % name)
    #     )
    # auc_df.to_csv(os.path.join(args.save_dir, "auc.csv"))
    # # 这里保存的是原始数据
    # for i, attr_name in enumerate(["trueD", "trueH", "trueR"]):
    #     save_arr = getattr(dataset, attr_name)
    #     pd.DataFrame(
    #         save_arr,
    #         columns=dataset.regions,
    #         index=dataset.epi_times.str
    #     ).to_csv(os.path.join(args.save_dir, "%s.csv" % attr_name))
    # 保存args到路径中（所有事情都完成再保存数据，安全）
    save_args = deepcopy(args.__dict__)
    save_args["model_type"] = "NetSEAIRQ-piecewise"
    utils.save(save_args, os.path.join(args.save_dir, "args.json"), "json")


if __name__ == "__main__":
    main()

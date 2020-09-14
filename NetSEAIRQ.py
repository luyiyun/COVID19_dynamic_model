import os
from copy import deepcopy
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
        De=5, Dq=14, c=13.0046, beta=2.03e-9, q=0,
        rho=0.6834, deltaI=0.1328, deltaQ=0.1259, gammaI=0.1029, gammaA=0.2978,
        gammaH=0.1024, theta=1.6003, nu=1.5008, phi=0.9,
        score_type="mse",
    ):
        """
        y0 = [H R E A I Sq Eq] + [S]
        """
        super().__init__(
            populations, y0for1, protect, protect_args, gamma_func_kwargs,
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
        for i in y0for1:
            y0s_i = np.r_[i, np.zeros(self.num_regions-1)]
            y0s_remain = y0s_remain - y0s_i
            y0s.append(y0s_i)
        y0s.append(y0s_remain)
        self.y0 = np.concatenate(y0s)

        # 在一级响应之前，允许I的流动，之后截断其流动
        self.I_flow_func = lambda t: t < self.protect_args["t0"]

        self._needRh=False

    def __call__(self, t, ALL):
        """
        t是时间参数，
        y0 = [H R E A I Sq Eq] + [S]
        y0 = [H R E A I Sq Eq] + [S] + [Rh]
        """
        HH = ALL[:self.num_regions]
        RR = ALL[self.num_regions:2*self.num_regions]
        EE = ALL[2*self.num_regions:3*self.num_regions]
        AA = ALL[3*self.num_regions:4*self.num_regions]
        II = ALL[4*self.num_regions:5*self.num_regions]
        SSq = ALL[5*self.num_regions:6*self.num_regions]
        EEq = ALL[6*self.num_regions:7*self.num_regions]
        SS = ALL[7*self.num_regions:8*self.num_regions]
        Nt = HH + RR + EE + AA + II + SSq + EEq + SS

        # 如果有保护措施，这里计算受到保护措施影响的c和q
        # decayc, decayq = self.protect_decay(t, **self.protect_args)
        if self.protect:
            decayc, decayq = self.protect_decay(t, **self.protect_args)
            ci, qi = self.c * decayc, self.q * decayq  # + 效果不好，还是用*吧
            # ci, qi = self.c * decayc, 0
        else:
            # ci, qi = self.c, self.q * decayq
            ci, qi = self.c, 0  # 完全没有控制的情况下，隔离率是0

        # 计算IAES的移动人口补充数量
        PmnT = self.PmnFunc(t) - np.diag(np.ones(self.num_regions))
        GammaT = self.GammaFunc(t)
        SS_out = (SS * GammaT).dot(PmnT)
        II_out = (II * GammaT).dot(PmnT)
        EE_out = (EE * GammaT).dot(PmnT)
        AA_out = (AA * GammaT).dot(PmnT) * self.I_flow_func(t)

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
        if self._needRh:
            RRh_ = self.gammaH*HH
            output = np.r_[output, RRh_]

        return output

    def predict(self, times, return_Rh=False):

        if return_Rh:
            self.y0 = np.r_[self.y0, np.zeros(self.num_regions)]
            self._needRh = True
        ALL = super().predict(times)
        HH = ALL[:, :self.num_regions]
        RR = ALL[:, self.num_regions:2*self.num_regions]
        EE = ALL[:, 2*self.num_regions:3*self.num_regions]
        AA = ALL[:, 3*self.num_regions:4*self.num_regions]
        II = ALL[:, 4*self.num_regions:5*self.num_regions]
        SSq = ALL[:, 5*self.num_regions:6*self.num_regions]
        EEq = ALL[:, 6*self.num_regions:7*self.num_regions]
        SS = ALL[:, 7*self.num_regions:8*self.num_regions]
        if return_Rh:
            RRh = ALL[:, 8*self.num_regions:]
            self.y0 = self.y0[:-self.num_regions]
            self._needRh = False
            return HH, RR, EE, AA, II, SSq, EEq, SS, RRh
        return HH, RR, EE, AA, II, SSq, EEq, SS

    @staticmethod
    def protect_decay(t, t0, c_k, q_k):
        decayc = utils.protect_decay2(t, t0, c_k)
        decayq = 1 - utils.protect_decay2(t, t0, q_k)  # 隔离率是慢慢从0升上去的
        return decayc, decayq

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

    def R0s(self, ts, remove_q=False, remove_H=False):
        Preds = self.predict(np.array(ts))
        All = 0
        for pred in Preds:
            All += pred
        S = Preds[-1] / All
        R0s = np.zeros((len(ts), self.num_regions))
        qi_multiply = 0. if remove_q else 1.
        H_multiply = 0. if remove_H else 1.
        for i, t in enumerate(ts):
            decayc, decayq = self.protect_decay(t, **self.protect_args)
            ci, qi = self.c * decayc, self.q * decayq  # + 效果不好，还是用*吧
            qi = qi * qi_multiply
            phi = self.phi * H_multiply
            betaC = self.beta * ci
            part1 = betaC * (1 - qi) / (
                self.deltaI * phi + self.gammaI * (1 - phi)
            )
            part2 = betaC * (1 - self.rho) * (1 - qi) * self.theta / self.gammaA
            part3 = betaC * self.nu * (1 - qi) / self.sigma
            R0s[i, :] = (part1 + part2 + part3) * S[i, :]
        return R0s

    def R0(self, ts, remove_q=False, remove_H=False):
        Preds = self.predict(np.array(ts))
        All = 0
        for pred in Preds:
            All += pred
        S = Preds[-1] / All
        R0s = np.zeros(len(ts))
        qi_multiply = 0. if remove_q else 1.
        H_multiply = 0. if remove_H else 1.

        sigma_1_rho_mat = np.diag(
            np.full(self.num_regions, self.sigma * (1 - self.rho))
        )
        sigma_rho_mat = np.diag(
            np.full(self.num_regions, self.sigma * self.rho)
        )

        for i, t in enumerate(ts):
            decayc, decayq = self.protect_decay(t, **self.protect_args)
            ci, qi = self.c * decayc, self.q * decayq  # + 效果不好，还是用*吧
            qi = qi * qi_multiply
            phi = self.phi * H_multiply
            # 得到一个三项都需要的乘数
            csq_ = np.diag(ci * (1 - qi) * S[i, :])
            # 计算M‘
            M_ = self.PmnFunc(t) - np.diag(np.ones(self.num_regions))
            M_ = M_.T * self.GammaFunc(t).reshape(1, self.num_regions)
            # 计算3个逆
            sigma1M = np.diag(np.full(self.num_regions, self.sigma)) - M_
            sigma1M_inv = np.linalg.inv(sigma1M)
            gammaA1M = np.diag(np.full(self.num_regions, self.gammaA)) - M_
            gammaA1M_inv = np.linalg.inv(gammaA1M)
            phi_I = phi * self.deltaI + (1 - phi) * self.gammaI
            phi1M = np.diag(np.full(self.num_regions, phi_I)) - M_
            phi1M_inv = np.linalg.inv(phi1M)
            # 计算3部分矩阵
            mat1 = self.nu * np.matmul(csq_, sigma1M_inv)
            mat2 = self.theta * np.matmul(csq_, gammaA1M_inv)
            mat2 = np.matmul(mat2, sigma_1_rho_mat)
            mat2 = np.matmul(mat2, sigma1M_inv)
            mat3 = np.matmul(csq_, phi1M_inv)
            mat3 = np.matmul(mat3, sigma_rho_mat)
            mat3 = np.matmul(mat3, sigma1M_inv)

            #
            mat = (mat1 + mat2 + mat3) * self.beta
            eigs, _ = np.linalg.eig(mat)
            R0s[i] = np.abs(eigs).max()

        return R0s

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
        params["y0for1[2:4]"] = (2, 0, 10)
        params["c"] = (1, 0, 100)
        params["beta"] = (1, 0, 1)
        params["q"] = (1, 0, 1)
        params["rho"] = (1, 0, 1)
        params["deltaI"] = (1, 0, 1)
        params["deltaQ"] = (1, 0, 1)
        params["gammaI"] = (1, 0, 1)
        params["gammaA"] = (1, 0, 1)
        params["gammaH"] = (1, 0, 1)
        params["phi"] = (1, 0, 1)
        params["theta"] = (1, 0, 10)
        params["nu"] = (1, 0, 10)
        params["protect_args-c_k"] = (31, 0, 1)
        params["protect_args-q_k"] = (31, 0, 1)
        # params["protect_args-c_k[4]"] = (1, 0, 1)  # 安徽
        # params["protect_args-c_k[21]"] = (1, 0, 1)  # 吉林
        # params["protect_args-q_k[4]"] = (1, 0, 1)  # 安徽
        # params["protect_args-q_k[21]"] = (1, 0, 1)  # 吉林
        return params


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
    parser.add_argument("--protect_ck", default=0.0, type=float)
    parser.add_argument("--protect_qk", default=0.0, type=float)
    parser.add_argument("--use_19", action="store_true")
    parser.add_argument("--zero_spring", action="store_true")
    parser.add_argument("--prophetPop", default=None, help="模板结果")
    args = parser.parse_args()  # 对于一些通用的参数，这里已经进行整理了

    """ 读取准备好的数据 """
    dat_file = "./DATA/Provinces.pkl"
    dataset = utils.Dataset(dat_file, args.t0, args.tm, args.fit_time_start)

    """ 构建、或读取、或训练模型 """
    # 根据不同的情况来得到合适的模型
    if args.model is not None:
        model = NetSEAIRQ.load(args.model)
    else:
        model = NetSEAIRQ(
            populations=dataset.populations,
            y0for1=np.array([0, 0, 0, 0, args.y0, 0, 0]),
            protect=True, score_type=args.fit_score,
            protect_args={
                "t0": dataset.protect_t0.delta,
                "c_k": args.protect_ck,
                "q_k": args.protect_qk
            },
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
    # ah_ind = dataset.regions.index("安徽")
    # jl_ind = dataset.regions.index("吉林")
    # import ipdb; ipdb.set_trace()
    # model.protect_args["c_k"][xj_ind] = 0.75
    if args.fit:
        # 设置我们拟合模型需要的数据
        if args.mask:
            use_ntime = (
                (dataset.epi_times - dataset.fit_start_t) >= 0).sum()
            mask = np.full((use_ntime, dataset.num_regions), True, np.bool)
            mask[:20, 0] = False  # hb
            mask[:28, 22] = False  # sd
        else:
            mask = None

        fit_start_index = (dataset.fit_start_t.ord - dataset.epi_t0.ord)
        fit_start_index = int(fit_start_index)
        score_kwargs = {
            "times": dataset.epi_times.delta[fit_start_index:],
            "mask": mask,
        }
        score_kwargs["trueH"] = dataset.trueH
        # score_kwargs["trueR"] = dataset.trueD + dataset.trueR
        # 搜索
        if args.fit_method == "annealing":
            fit_kwargs = {
                "callback": utils.callback, "method": "annealing"
            }
        else:
            if args.prophetPop is not None:
                prophet_opt = utils.load(args.prophetPop)
                # prophetPop = prophet_opt["population"]
            else:
                prophet_opt = None
            fit_kwargs = {
                "method": args.fit_method,
                "fig_dir": args.save_dir+"/",
                "njobs": -1,
                "NIND": args.geatpy_nind,
                "MAXGEN": args.geatpy_maxgen,
                "n_populations": args.geatpy_npop,
                "opt_res": prophet_opt
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
        utils.save(opt_res, os.path.join(args.save_dir, "opt_res.pkl"))
    model.save(os.path.join(args.save_dir, "model.pkl"))

    # 预测结果
    prot_preds = model.predict(dataset.pred_times.delta, return_Rh=True)
    model.protect = False
    nopr_preds = model.predict(dataset.pred_times.delta, return_Rh=True)
    model.protect = True
    model.lam = 1 / 28
    prot_preds_28 = model.predict(dataset.pred_times.delta)

    """ 计算相关指标以及绘制图像 """
    # 预测R0
    pass

    # 计算每个地区的曲线下面积以及面积差,并保存
    auc = under_area(
        dataset.epi_times.delta,
        dataset.trueH,
        dataset.pred_times.delta,
        nopr_preds[0],
    )
    auc_df = pd.DataFrame(
        auc.T, columns=["true_area", "pred_area", "diff_area"],
        index=dataset.regions
    )
    auc_df["population"] = dataset.populations
    auc_df["diff_norm"] = auc_df.diff_area / auc_df.population
    auc_df.sort_values("diff_norm", inplace=True)

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
                # ----- 真实疫情发展
                ("trueH", dataset.epi_times.ord.astype("int"),
                 dataset.trueH[:, i], "ro"),
                # ("trueR", dataset.epi_times.ord.astype("int"),
                #  dataset.trueR[:, i]+dataset.trueD[:, i], "bo"),
                # ----- 预测
                ("predH", dataset.pred_times.ord.astype("int"),
                 prot_preds[0][:, i], "r"),
                # ("predR", dataset.pred_times.ord.astype("int"),
                #  prot_preds[1][:, i], "b"),
                ("predE", dataset.pred_times.ord.astype("int"),
                 prot_preds[3][:, i], "y"),
                ("predA", dataset.pred_times.ord.astype("int"),
                 prot_preds[4][:, i], "g"),
                ("predI", dataset.pred_times.ord.astype("int"),
                 prot_preds[4][:, i], "c"),
                # ----- 将隔离延长找28天
                ("predH", dataset.pred_times.ord.astype("int"),
                 prot_preds_28[0][:, i], "r--"),
                # ("predR", dataset.pred_times.ord.astype("int"),
                #  prot_preds_28[1][:, i], "b"),
                ("predE", dataset.pred_times.ord.astype("int"),
                 prot_preds_28[3][:, i], "y--"),
                ("predA", dataset.pred_times.ord.astype("int"),
                 prot_preds_28[4][:, i], "g--"),
                ("predI", dataset.pred_times.ord.astype("int"),
                 prot_preds_28[4][:, i], "c--"),
            ],
            [
                ("trueH", dataset.epi_times.ord.astype("int"),
                 dataset.trueH[:, i], "ro"),
                # ("trueR", dataset.epi_times.ord.astype("int"),
                #  dataset.trueR[:, i]+dataset.trueD[:, i], "bo"),
                ("predH", dataset.pred_times.ord.astype("int"),
                 nopr_preds[0][:, i], "r"),
                # ("predR", dataset.pred_times.ord.astype("int"),
                #  nopr_preds[1][:, i], "b"),
                ("predE", dataset.pred_times.ord.astype("int"),
                 nopr_preds[3][:, i], "y"),
                ("predA", dataset.pred_times.ord.astype("int"),
                 nopr_preds[4][:, i], "g"),
                ("predI", dataset.pred_times.ord.astype("int"),
                 nopr_preds[4][:, i], "c"),
            ],
            save_dir=img_dir
        )
    # 保存结果
    for i, name in enumerate([
        "predH", "predR", "predE", "predA", "predI",
        "predSq", "predEq", "predS", "predRh"
    ]):
        pd.DataFrame(
            prot_preds[i],
            columns=dataset.regions,
            index=dataset.pred_times.str
        ).to_csv(
            os.path.join(args.save_dir, "protect_%s.csv" % name),
            encoding="utf_8_sig"
        )
        pd.DataFrame(
            nopr_preds[i],
            columns=dataset.regions,
            index=dataset.pred_times.str
        ).to_csv(
            os.path.join(args.save_dir, "noprotect_%s.csv" % name),
            encoding="utf_8_sig"
        )
    auc_df.to_csv(os.path.join(args.save_dir, "auc.csv"), encoding="utf_8_sig")
    # 这里保存的是原始数据
    for i, attr_name in enumerate(["trueD", "trueH", "trueR"]):
        save_arr = getattr(dataset, attr_name)
        pd.DataFrame(
            save_arr,
            columns=dataset.regions,
            index=dataset.epi_times.str
        ).to_csv(os.path.join(args.save_dir, "%s.csv" % attr_name),
                 encoding="utf_8_sig")
    # 保存args到路径中（所有事情都完成再保存数据，安全）
    save_args = deepcopy(args.__dict__)
    save_args["model_type"] = "NetSEAIRQ"
    utils.save(save_args, os.path.join(args.save_dir, "args.json"), "json")


if __name__ == "__main__":
    main()

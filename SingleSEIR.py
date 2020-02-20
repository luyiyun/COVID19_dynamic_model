import os

import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from sko import Real

import utils
from model import InfectiousBase
from plot import plot_one_regions


class NetSEIR(InfectiousBase):
    def __init__(
        self, De, Di, y0, alpha_I=None,
        alpha_E=None, protect=False, protect_args=None,
        num_people=None, score_type="mse", fit_method="scipy-NM"
    ):
        super().__init__(
            De, Di, y0, alpha_I, alpha_E,
            protect, protect_args, num_people, score_type, fit_method
        )
        if self.protect and self.protect_args is None:
            raise ValueError("protect_args must not be None!")

        # 实际使用的是theta和beta，所以De和Di是不能作为参数来fit的
        self.theta = 1 / self.De
        self.beta = 1 / self.Di

    def __call__(self, t, SEI):
        """
        t是时间参数，
        SEI分别是各个地区的S、各个地区的E、各个地区的I组成的一维ndarray向量
        """
        SS = SEI[0]
        EE = SEI[1]
        II = SEI[2]

        if self.protect:
            alpha_E = self.alpha_E * self.protect_decay(t, **self.protect_args)
            alpha_I = self.alpha_I * self.protect_decay(t, **self.protect_args)
        else:
            alpha_E, alpha_I = self.alpha_E, self.alpha_I

        s2e_i = alpha_I * SS * II
        s2e_e = alpha_E * EE * SS
        e2i = self.theta * EE
        i2r = self.beta * II

        delta_s = - s2e_i - s2e_e
        delta_e = s2e_e + s2e_i - e2i
        delta_i = e2i - i2r
        output = np.r_[delta_s, delta_e, delta_i]
        return output

    def predict(self, times):
        SEI = super().predict(times)
        if self.num_people is not None:
            SEI = SEI * self.num_people
        return SEI

    @staticmethod
    def protect_decay(t, t0, k):
        """ 最简单的一种控制措施 """
        return np.exp(-k * (t - t0))

    def score(self, times, true_infects):
        """
        因为这里是多个地区的预测，所以true_infects也是一个二维矩阵，即
        shape = num_times x num_regions
        """
        _, _, preds = self.predict(times)
        if self.score_type == "mse":
            return np.mean((true_infects - preds) ** 2)
        elif self.score_type == "nll":
            return np.mean(preds - np.log(preds) * true_infects)
        else:
            raise ValueError

    def _set_args(self, args):
        self.alpha_E = self.kwargs["alpha_E"] = args[0]
        self.alpha_I = self.kwargs["alpha_I"] = args[1]
        self.protect_args["k"] = self.kwargs["protect_args"]["k"] = args[2]

    def _get_fit_x0(self):
        if self.fit_method == "scipy-NM":
            return [0.5, 0.5, 0.5]
        elif self.fit_method == "scikit-optimize":
            return [Real(0, 2), Real(0, 2), Real(0, 100)]
        elif self.fit_method == "scikit-opt":
            return [3, np.array([0, 0, 0]), np.array([2, 2, 100])]


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--save_dir")
    parser.add_argument("--region_type", default="city",
                        choices=("city", "province"))
    parser.add_argument(
        "--model", default=None,
        help=("默认是None，即不使用训练的模型，而是直接使用命令行赋予的参数"
              "，不然则读取拟合的参数，命令行赋予的参数无效")
    )
    parser.add_argument(
        "--regions", default=None, nargs="+",
        help=("默认是None，则对于省份或城市都使用不同的默认值，不然，则需要键入需要估计"
              "的地区名。如果是all，则将所有的都计算一下看看")
    )
    parser.add_argument("--t0", default="2019-12-01", help="疫情开始的那天")
    parser.add_argument("--y0", default=1, type=float,
                        help="武汉或湖北在t0那天的感染人数")
    parser.add_argument("--tm", default="2020-02-15", help="需要预测到哪天")
    parser.add_argument("--alpha_E", default=0.15, type=float)
    parser.add_argument("--alpha_I", default=0.15, type=float)
    parser.add_argument("--De", default=5.2, type=float)
    parser.add_argument("--Di", default=15, type=float)
    parser.add_argument("--protect_t0", default="2020-01-23")
    parser.add_argument("--protect_k", default=1., type=float)
    # parser.add_argument("--protect_eta", default=0.89, type=float)
    # parser.add_argument("--protect_tm", default="2020-02-01")
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
    protect_t0 = utils.time_str2diff(args.protect_t0, args.t0)
    # protect_tm = utils.time_str2diff(args.protect_tm, args.t0)
    tm = utils.time_str2diff(args.tm, args.t0)

    # 保存args到路径中
    utils.save(args.__dict__, os.path.join(save_dir, "args.json"), "json")

    # 读取准备好的数据
    if args.region_type == "city":
        dat_file = "./DATA/City.pkl"
    else:
        dat_file = "./DATA/Provinces.pkl"
    all_dat = utils.load(dat_file, "pkl")
    dat_t0 = all_dat["t0"]
    regions = all_dat["regions"]
    num_times = all_dat["num_times"]
    epidemic = all_dat["epidemic"]
    population = all_dat["population"]
    # 对这些数据进行整理
    t0_ord = utils.time_str2ord(args.t0)
    dat_t0_ord = utils.time_str2ord(dat_t0)
    dat_t0_diff = utils.time_str2diff(dat_t0, args.t0)
    pmn = {(k + dat_t0_ord - t0_ord): v
           for k, v in pmn.items()}
    epidemic_times = np.array([i+dat_t0_ord-t0_ord for i in range(num_times)])
    pred_times = np.arange(0, tm)
    # 为每个地区安排不同的开始时间
    pass

    # 构建模型
    if args.region_type == "city":
        y0 = np.array([args.y0 if pr == "武汉" else 0 for pr in regions])
    else:
        y0 = np.array([args.y0 if ci == "湖北" else 0 for ci in regions])
        # alpha_E = np.array([args.alpha_E * 2 if ci == "湖北" else args.alpha_E
        #                     for ci in regions])
        # alpha_I = np.array([args.alpha_I * 2 if ci == "湖北" else args.alpha_I
        #                     for ci in regions])
        alpha_E, alpha_I = args.alpha_E, args.alpha_I
    y0 = y0 / population
    y0 = np.r_[np.ones(len(regions)), np.zeros(len(regions)), y0]
    if args.model is not None:
        raise ValueError
        model = NetSEIR.load(os.path.join(save_dir, "model.pkl"))
    else:
        # 预测结果
        model = NetSEIR(
            args.De, args.Di, y0, (protect_t0, protect_tm), (pmn,),
            alpha_I, alpha_E, protect=True, num_people=population,
            protect_args={"t0": protect_t0,
                          "tm": protect_tm,
                          "eta": args.protect_eta}
        )
        _, pred_EE_prot, pred_II_prot = model.predict(pred_times)
        model.protect = False
        _, pred_EE_nopr, pred_II_nopr = model.predict(pred_times)

    # 先得到要画的地区的索引
    if plot_regions[0] == "all":
        plot_regions = [(i, reg) for i, reg in enumerate(regions)]
    else:
        plot_regions = [(regions.index(reg), reg) for reg in plot_regions]
    # 绘制每个地区的图片，并保存
    plt.rcParams['font.sans-serif'] = ['SimHei']
    for i, reg in plot_regions:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        print("%d: %s" % (i, reg))
        plot_one_regions(
            axes[0], pred_times, epidemic_times, pred_EE_nopr[:, i],
            pred_II_nopr[:, i], epidemic[:, i], reg+" no protect", t0=args.t0
        )
        plot_one_regions(
            axes[1], pred_times, epidemic_times, pred_EE_prot[:, i],
            pred_II_prot[:, i], epidemic[:, i], reg+" protect", t0=args.t0
        )
        fig.savefig(os.path.join(save_dir, "%s.png" % reg))

    # # 参数设置
    # region_type, dir_name = sys.argv[1:3]
    # De = 3
    # Di = 9
    # protect_t0 = "2020-01-23"  # 武汉封城的时间
    # save_dir = os.path.join("./RESULTS/", dir_name)
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)

    # # 读取疫情数据
    # if region_type == "province":
    #     dat_file = "./DATA/Provinces.pkl"
    # elif region_type == "city":
    #     dat_file = "./DATa/City.pkl"
    # else:
    #     raise ValueError
    # with open(dat_file, "rb") as f:
    #     all_dat = pickle.load(f)
    # t0 = all_dat["t0"]
    # regions = all_dat["regions"]
    # num_times = all_dat["num_times"]
    # pmn = all_dat["pmn"]
    # epidemic = all_dat["epidemic"]
    # population = all_dat["population"]

    # # 模型
    # num_regions = len(regions)
    # y0 = epidemic[0, :] / population
    # y0 = np.r_[np.ones(num_regions), np.zeros(num_regions), y0]
    # model = NetSEIR(
    #     De, Di, y0, (), (pmn,), protect=True,
    #     protect_args={"t0": utils.time_str2diff(protect_t0, t0),
    #                   "tm": utils.time_str2diff("2020-02-11", t0),
    #                   "eta": 0.89},
    #     num_people=population, score_type="mse",
    #     fit_method="scikit-opt"
    # )
    # t1 = perf_counter()
    # model.fit(np.arange(num_times), epidemic)
    # t2 = perf_counter()
    # print(t2-t1)

    # # 保存
    # model.save(os.path.join(save_dir, "model.pkl"))

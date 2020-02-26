import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from scipy.integrate import trapz
# from scipy.optimize import dual_annealing

import utils
from geatyAlg import geaty_func
from plot import plot_one_regions
from NetSEAIRQ import NetSEAIRQ, score_func


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
    # parser.add_argument("--y0", default=100, type=float,
    #                     help="武汉或湖北在t0那天的感染人数")
    parser.add_argument("--tm", default="2020-03-31", help="需要预测到哪天")
    # =0并且不进行训练，则模型认为潜伏期没有传染性
    # parser.add_argument("--alpha_E", default=0.0, type=float)
    # parser.add_argument("--alpha_I", default=0.15, type=float)
    parser.add_argument("--De", default=5, type=float)
    parser.add_argument("--Dq", default=14, type=float)
    # parser.add_argument("--protect_t0", default="2020-01-23")
    parser.add_argument("--protect_k", default=0.001, type=float)
    parser.add_argument("--fit_score", default="nll", choices=["nll", "mse"])
    parser.add_argument("--fit_time_start", default="2020-02-01")
    # parser.add_argument("--use_pmn_mean", action="store_true")
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
    protect_t0_relative = dats["response_time"] - t0  # 防控开始时间
    # protect_t0_relative = utils.time_str2diff(args.protect_t0, args.t0)
    epi_t0_relative = dats["epidemic_t0"] - t0  # 第一个确诊病例出现的时间
    pmn_matrix_relative = {(k-t0): v for k, v in dats["pmn"].items()}  # pmn的时间
    epi_times_relative = np.arange(  # 确诊病例时间
        epi_t0_relative, epi_t0_relative + dats["trueH"].shape[0]
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

    """ 构建、或读取、或训练模型 """
    # 根据不同的情况来得到合适的模型
    if args.model is not None and args.model != "fit":
        model = NetSEAIRQ.load(args.model)
    else:
        model = NetSEAIRQ(
            populations=dats["population"], t0=t0,
            y0for1=[41, 2, 0, 0, 0, 0, 0, 739], protect=True,
            protect_args={"t0": protect_t0_relative, "c_k": 0.0, "q_k": 0.0},
            GammaFuncArgs={"gammas": 0.1255,
                           "protect_t0": protect_t0_relative},
            PmnFuncArgs={"pmn": pmn_matrix_relative, "use_mean": False},
            De=args.De, Dq=args.Dq, score_type=args.fit_score,
        )
        if args.model == "fit":
            # 设置我们拟合模型需要的数据
            mask = np.full(num_regions, 1)
            mask[hb_wh_index] = 0
            use_fit_data_start = utils.time_str2ord(args.fit_time_start) - \
                dats["epidemic_t0"]
            use_fit_data_epid = {
                "times": epi_times_relative[use_fit_data_start:],
                "trueH": dats["trueH"][use_fit_data_start:],
                "trueR": dats["trueR"][use_fit_data_start:],
                "trueD": dats["trueD"][use_fit_data_start:],
            }

            # 搜索
            def func(x):
                return score_func(x, model, mask=mask, **use_fit_data_epid)
            dim, lb, ub = model.params_fit_range()
            opt_res = geaty_func(
                func, dim=dim, lb=lb, ub=ub,
                Encoding="BG", NIND=400, MAXGEN=25,
                fig_dir=save_dir+"/", njobs=-1
            )
            # opt_res_1 = dual_annealing(
            #     func, np.stack([lb, ub], axis=1), maxiter=10000,
            #     callback=utils.callback
            # )
            # opt_res = {"all": opt_res_1, "BestParam": opt_res_1.x}

            # 把拟合得到的参数整理成dataframe，然后打印一下
            pd.DataFrame(dict(params=opt_res["BestParam"])).to_csv(
                os.path.join(save_dir, "params.csv")
            )

            # 将得到的最优参数设置到模型中，并保存
            model.set_params(opt_res["BestParam"])
            model.save(os.path.join(save_dir, "model.pkl"))
            utils.save(opt_res, os.path.join(save_dir, "opt_res.pkl"), "pkl")

    # 预测结果
    predH_prot = model.predict(pred_times_relative)[0]
    model.protect = False
    predH_nopr = model.predict(pred_times_relative)[0]

    """ 计算相关指标以及绘制图像 """
    # 预测R0
    # R0s = model.R0(pred_times_relative)
    # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    # ax.plot(pred_times_relative, R0s)
    # fig.savefig(os.path.join(save_dir, "R0.png"))
    # plt.close(fig)
    # 首先把相关结果数据都汇集
    results = {
        "pred_times": pred_times_relative + t0,
        # "pred_EE": {"no_protect": pred_EE_nopr, "protect": pred_EE_prot},
        "predH": {"no_protect": predH_nopr, "protect": predH_prot},
        "true_times": epi_times_relative + t0,
        "trueH": dats["trueH"],
    }
    # 计算每个地区的曲线下面积以及面积差
    predH_nopr_only_true = predH_nopr[  # true只是pred的一部分
        np.isin(pred_times_relative, epi_times_relative), :]
    true_areas, pred_areas, diff_areas = [], [], []
    for i, region in enumerate(dats["regions"]):
        true_area = trapz(epi_times_relative, dats["trueH"][:, i])
        pred_area = trapz(epi_times_relative, predH_nopr_only_true[:, i])
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
        print("%d: %s" % (i, reg))
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        plot_one_regions(
            axes[0], pred_times_relative, epi_times_relative,
            predH_nopr[:, i], dats["trueH"][:, i],
            reg+" no protect", t0_ord=t0
        )
        plot_one_regions(
            axes[1], pred_times_relative, epi_times_relative,
            predH_prot[:, i], dats["trueH"][:, i],
            reg+" protect", t0_ord=t0
        )
        plot_one_regions(
            axes[2], epi_times_relative, epi_times_relative,
            predH_nopr[time_mask, i], dats["trueH"][:, i],
            reg+" no protect part", t0_ord=t0, use_log=True
        )
        fig.savefig(os.path.join(save_dir, "part1/%s.png" % reg))
        plt.close(fig)
        # 另一部分
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes = axes.flatten()
        plot_one_regions(
            axes[0], pred_times_relative, epi_times_relative,
            predH_nopr[:, i], dats["trueH"][:, i],
            reg+" no protect", t0_ord=t0
        )
        plot_one_regions(
            axes[1], epi_times_relative, epi_times_relative,
            predH_nopr[time_mask, i],
            dats["trueH"][:, i], reg+" no protect part", t0_ord=t0,
            use_log=True
        )
        fig.savefig(os.path.join(save_dir, "part2/%s.png" % reg))
        plt.close(fig)

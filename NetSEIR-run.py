import os
import os.path as osp
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import utils
from plot import under_area, plot_one_regions
from NetSEIR import NetSEIR


def mk_dir(root_dir, new_dir):
    if not osp.exists(root_dir):
        os.mkdir(root_dir)
    full_dir = osp.join(root_dir, new_dir)
    if not osp.exists(full_dir):
        os.mkdir(full_dir)
    return full_dir


def main():

    """ Simulation Parameters """
    Dis = [11.5]
    Des = [5.2]
    y0s = [1]
    out_rates = ["19", "20"]  # ["19", "20"]
    outIsToZeroInSprings = [True, False]
    R0s = np.linspace(2.2, 6.47, 5)  # np.linspace(2.2, 6.47, 10)
    ks = [0, 0.25, 0.5]  # np.arange(0, 5, 0.5)

    """ Read Data and Prepare """
    root_dir = "./RESULTS/test0329"
    dat_file = "./DATA/Provinces.pkl"
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    dataset = utils.Dataset(
        dat_file, t0="2019-12-01", tm="2020-04-30",
        fit_start_t="2020-01-26"
    )

    """ Iteration """
    for params in product(
        out_rates, outIsToZeroInSprings, Des, Dis, R0s, ks, y0s
    ):
        out_rate, outIsToZeroInSpring, De, Di, R0, k, y0 = params
        # 创建保存路径
        save_name = (
            "out_rate=%s-SpringZero=%s-De=%s-Di=%s-R0=%s-k=%s-y0=%s" %
            tuple(["%.2f" % p if isinstance(p, float) else str(p)
                   for p in params])
        )
        save_dir = mk_dir(root_dir, save_name)
        # 使用不同年份的迁出率
        out_dict = getattr(dataset, "out%s_dict" % out_rate)
        # 是否春节期间迁出率设为0
        if outIsToZeroInSpring:
            zero_period = [tm.relative for tm in dataset.zero_period]
        else:
            zero_period = None
        # 使用不同的潜伏期
        # 使用不同的传染期
        # 使用不同的R0
        alpha_I = R0 / Di
        # 使用不同的k
        print(save_name)
        # 建立模型
        model = NetSEIR(
            De=De, Di=Di, populations=dataset.populations,
            y0for1=y0, alpha_I=alpha_I, alpha_E=0.0, protect=True,
            protect_args={"t0": dataset.protect_t0.relative, "k": k},
            gamma_func_kwargs={"gammas": out_dict, "zero_period": zero_period},
            Pmn_func_kwargs={"pmn": dataset.pmn_matrix_relative}
        )
        # 预测结果
        prot_preds = model.predict(dataset.pred_times.relative)
        model.protect = False
        nopr_preds = model.predict(dataset.pred_times.relative)
        # 计算每个地区的曲线下面积以及面积差,并保存
        auc = under_area(
            dataset.epi_times.relative, dataset.trueH,
            dataset.pred_times.relative, nopr_preds[2]
        )
        auc_df = pd.DataFrame(
            auc.T, columns=["true_area", "pred_area", "diff_area"],
            index=dataset.regions
        )
        auc_df["population"] = dataset.populations
        auc_df["diff_norm"] = auc_df.diff_area / auc_df.population
        auc_df.sort_values("diff_norm", inplace=True)
        # 为每个地区绘制曲线图
        img_dir = osp.join(save_dir, "imgs")
        if not osp.exists(img_dir):
            os.mkdir(img_dir)
        for i, reg in enumerate(dataset.regions):
            plot_one_regions(
                reg,
                [
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
        # 保存结果
        for i, name in enumerate(["predS", "predE", "predI", "predR"]):
            pd.DataFrame(
                prot_preds[i],
                columns=dataset.regions,
                index=dataset.pred_times.str
            ).to_csv(
                osp.join(save_dir, "protect_%s.csv" % name)
            )
            pd.DataFrame(
                nopr_preds[i],
                columns=dataset.regions,
                index=dataset.pred_times.str
            ).to_csv(
                osp.join(save_dir, "noprotect_%s.csv" % name)
            )
        auc_df.to_csv(osp.join(save_dir, "auc.csv"))
        # 这里保存的是原始数据，保存到root_dir下
        for i, attr_name in enumerate(["trueD", "trueH", "trueR"]):
            save_arr = getattr(dataset, attr_name)
            pd.DataFrame(
                save_arr,
                columns=dataset.regions,
                index=dataset.epi_times.str
            ).to_csv(osp.join(root_dir, "%s.csv" % attr_name))


if __name__ == "__main__":
    main()

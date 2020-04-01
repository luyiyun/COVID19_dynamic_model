from os.path import join
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

import utils


def main():
    save_dir = sys.argv[1]
    auc = utils.load(join(save_dir, "auc.pkl"))
    params = pd.read_csv(join(save_dir, "params.csv"))
    dats = utils.load("./DATA/Provinces.pkl")

    # auc面积总结
    summ = pd.DataFrame({
        "regions": dats["regions"],
        "auc_diff": auc[2, :],
        "auc_norm": auc[2, :] / dats["population"],
        "k": params.iloc[2:-1, 1],
        "population": dats["population"]
    })
    summ.sort_values("auc_norm", inplace=True)
    print(summ)
    summ.to_csv(join(save_dir, "summary.csv"))

    # 人流量总结
    temp_dict = {}
    sum_all = 0
    for i, (k, v) in enumerate(dats["pmn"].items()):
        temp_arr = dats["out_trend20"][i, :] * dats["population"]
        temp_mat = np.diag(temp_arr)
        temp_mat = np.matmul(temp_mat, dats["pmn"][k])
        temp_dict[k] = temp_mat
        sum_all += temp_mat
    mean_all = sum_all / len(temp_dict)

    plt.rcParams["font.sans-serif"] = ["SimHei"]
    # 非对数尺度
    fig, ax = plt.subplots()
    im = ax.imshow(mean_all, cmap=plt.cm.hot_r)
    ax.set_yticks(range(len(dats["regions"])))
    ax.set_xticks(range(len(dats["regions"])))
    ax.set_yticklabels(dats["regions"])
    ax.set_xticklabels(dats["regions"], rotation=90)
    fig.colorbar(im)
    fig.savefig(join(save_dir, "out_heatmap.png"))
    # 对数尺度
    fig, ax = plt.subplots()
    im = ax.imshow(mean_all, cmap=plt.cm.hot_r,
                   norm=matplotlib.colors.LogNorm())
    ax.set_yticks(range(len(dats["regions"])))
    ax.set_xticks(range(len(dats["regions"])))
    ax.set_yticklabels(dats["regions"])
    ax.set_xticklabels(dats["regions"], rotation=90)
    fig.colorbar(im)
    fig.savefig(join(save_dir, "out_heatmap_log.png"))


if __name__ == "__main__":
    main()
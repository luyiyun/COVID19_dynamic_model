import os
from datetime import date

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz


def plot_one_regions(
    ax, pred_times, true_times, pred_I, true_I, region_name, use_log=False
):
    """
    将E和I、还有真实的感染人数绘制到一个axe上
    pred_times是预测的时间点，true_times是真实数据的时间点，
    region_name是本次使用的数据是哪个地区的
    """
    ax.plot(pred_times, pred_I, "-r", label="predict")
    ax.plot(true_times, true_I, "or", label="true")
    if use_log:
        ax.set_yscale("log")
    tick_time = [t for t in pred_times if t % 5 == 0]
    # tick_times_ord = [t+t0_ord for t in tick_time]
    tick_times_str = [str(date.fromordinal(t))[5:] for t in tick_time]
    ax.set_xticks(tick_time)
    ax.set_xticklabels(tick_times_str, rotation=45)
    ax.set_title(region_name)
    ax.legend()


def under_area(true_times, true_values, pred_times, pred_values):
    pred_part = pred_values[np.isin(pred_times, true_times), :]
    if true_values.ndim == 1:
        num_regions = 1
        true_values = np.expand_dims(true_values, axis=1)
        pred_values = np.expand_dims(pred_values, axis=1)
    elif true_values.ndim == 2:
        num_regions = true_values.shape[1]
    else:
        raise ValueError
    true_areas, pred_areas, diff_areas = [], [], []
    for i in range(num_regions):
        true_area = trapz(true_times, true_values[:, i])
        pred_area = trapz(true_times, pred_part[:, i])
        true_areas.append(true_area)
        pred_areas.append(pred_area)
        diff_areas.append(pred_area - true_area)
    return np.array([true_areas, pred_areas, diff_areas])


def plot_all(regions, results, save_dir, t0_ord):
    """ 利用预测结果来绘制每个地区的图像 """
    if regions[0] == "all":
        plot_regions = [(i, reg) for i, reg in enumerate(results["regions"])]
    else:
        plot_regions = [
            (results["regions"].index(reg), reg) for reg in plot_regions
        ]
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # time_mask = np.isin(results["pred_times"], results["true_times"])

    # 绘制每个地区的图片，并保存
    for i, reg in plot_regions:
        print("%d: %s" % (i, reg))
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        plot_one_regions(
            axes[0], results["pred_times"], results["true_times"],
            results["pred_nopr"][:, i], results["trueH"][:, i],
            reg+" no protect"
        )
        plot_one_regions(
            axes[1], results["pred_times"], results["true_times"],
            results["pred_prot"][:, i], results["trueH"][:, i],
            reg+" protect"
        )
        plot_one_regions(
            axes[2], results["pred_times"], results["true_times"],
            results["pred_nopr"][:, i], results["trueH"][:, i],
            reg+" no protect log", use_log=True
        )
        plot_one_regions(
            axes[3], results["pred_times"], results["true_times"],
            results["pred_prot"][:, i], results["trueH"][:, i],
            reg+" protect log", use_log=True
        )
        fig.savefig(os.path.join(save_dir, "%s.png" % reg))
        plt.close(fig)

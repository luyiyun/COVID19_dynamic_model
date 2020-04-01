import os
from datetime import date

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz


def under_area(true_times, true_values, pred_times, pred_values):
    """
    计算true line和pred line的重叠时间段的曲线下面积

    Arguments:
        true_times {ndarray} -- 真实曲线的时间点
        true_values {ndarray} -- 真实曲线的值, shape=(n_times_t, n_regions)
        pred_times {ndarray} -- 预测曲线的时间点
        pred_values {ndarray} -- 预测曲线的时间点, shpae=(n_times_p, n_regions)

    Raises:
        None

    Returns:
        ndarray -- shape=(3, n_regions)，分别是真实曲线下面积、预测曲线下面积，两者面积之差
    """
    comm_times, comm1, comm2 = np.intersect1d(
        true_times, pred_times, return_indices=True
    )
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
        true_area = trapz(comm_times, true_values[comm1, i])
        pred_area = trapz(comm_times, pred_values[comm2, i])
        true_areas.append(true_area)
        pred_areas.append(pred_area)
        diff_areas.append(pred_area - true_area)
    return np.array([true_areas, pred_areas, diff_areas])


def plot_one_axes(ax, name, *elems, use_log=False):
    """
    将多组数据绘制到以一个axes上

    Arguments:
        ax {Axes} -- 要绘制的axes对象
        name {str} -- 图片名称
        elems {tuples} -- 四元组，可以给出多个，每一组表示一条线段，依次是（label，x，y，fmt）

    Keyword Arguments:
        use_log {bool} -- 是否将y的坐标置换为对数坐标 (default: {False})
    """
    all_times = set()
    for label, x, y, fmt in elems:
        ax.plot(x, y, fmt, label=label)
        all_times.update(x)
    if use_log:
        ax.set_yscale("log")
    tick_time = [t for t in all_times if t % 5 == 0]
    tick_times_str = [str(date.fromordinal(t))[5:] for t in tick_time]
    ax.set_xticks(tick_time)
    ax.set_xticklabels(tick_times_str, rotation=45)
    ax.set_title(name)
    ax.legend()


def plot_one_regions(
    reg_name, protect_elems, noprotect_elems, save_dir=None
):
    """
    绘制一个地区的图像，需要提供protect和noprotect两个方面的数据

    Arguments:
        reg_name {str} -- 该地区的名称
        protect_elems {list of 4-tuples} -- protect方面的数据组成的tuples
        noprotect_elems {list of 4-tuples} -- no protect方面的数据组成的tuples

    Keyword Arguments:
        save_dir {str} -- 保存的路径，如果是None则不保存 (default: {None})
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    plot_one_axes(axes[0], "No Protect", *noprotect_elems)
    plot_one_axes(axes[1], "Protect", *protect_elems)
    plot_one_axes(axes[2], "No Protect (log)", *noprotect_elems, use_log=True)
    plot_one_axes(axes[3], "Protect (log)", *protect_elems, use_log=True)
    fig.suptitle(reg_name)
    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, "%s.png" % reg_name))
    else:
        plt.show()
    plt.close(fig)

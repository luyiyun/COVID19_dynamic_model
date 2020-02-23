from datetime import date


def plot_one_regions(
    ax, pred_times, true_times, pred_E, pred_I, true_I, region_name,
    t0_ord=None, use_log=False
):
    """
    将E和I、还有真实的感染人数绘制到一个axe上
    pred_times是预测的时间点，true_times是真实数据的时间点，
    region_name是本次使用的数据是哪个地区的
    """
    # ax.plot(pred_times, pred_E, "-y", label="predict")
    ax.plot(pred_times, pred_I, "-r", label="predict")
    ax.plot(true_times, true_I, "or", label="true")
    if use_log:
        ax.set_yscale("log")
    if t0_ord is not None:
        tick_time = [t for t in pred_times if t % 5 == 0]
        tick_times_ord = [t+t0_ord for t in tick_time]
        tick_times_str = [str(date.fromordinal(t))[5:] for t in tick_times_ord]
        ax.set_xticks(tick_time)
        ax.set_xticklabels(tick_times_str, rotation=45)
    ax.set_title(region_name)
    ax.legend()

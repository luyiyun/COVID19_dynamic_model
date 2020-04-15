import os.path as osp
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import load, Dataset
from NetSEIR import NetSEIR
from NetSEAIRQ import NetSEAIRQ


def main():
    # 读取数据
    root_dir = sys.argv[1]
    # dat = load("./DATA/Provinces.pkl", "pkl")
    args = load(osp.join(root_dir, "args.json"), "json")
    if args["model_type"] == "NetSEIR":
        model_class = NetSEIR
    elif args["model_type"] == "NetSEAIRQ":
        model_class = NetSEAIRQ
    else:
        raise NotImplementedError
    model = model_class.load(osp.join(root_dir, "model.pkl"))
    dataset = Dataset(
        "./DATA/Provinces.pkl", args["t0"], args["tm"],
        args["fit_time_start"]
    )

    if args["model_type"] == "NetSEIR":
        # 1. 整理params.csv
        new_index = ["k of %s" % i for i in dataset.regions]
        new_index = ["R0",  "y0 in 2019-12-01", "alpha_I", "alpha_E"] +\
            new_index
        new_values = np.r_[
            model.alpha_I * model.Di,
            model.y0for1,
            model.alpha_I, model.alpha_E,
            model.protect_args["k"]
        ]
        new_params = pd.Series(new_values, index=new_index)
        new_params.to_csv(osp.join(root_dir, "params_clear.csv"))

        # 2. 计算R0
        r0 = model.R0(dataset.pred_times.delta)
        r0_prot = model.R0(dataset.pred_times.delta, protect=True)
        r0_relative = model.R0(dataset.pred_times.delta, relative=True)
        rt = model.R0(dataset.pred_times.delta, protect=True, relative=True)
        fig, axs = plt.subplots(ncols=2)
        axs[0].plot(r0, c="r", label="no protect")
        axs[1].plot(r0_prot, c="b", label="protect")
        axs[0].plot(r0_relative, c="g", label="herd immunity")
        axs[1].plot(rt, c="black", label="rt")
        for ax in axs:
            ax.set_xticks(np.arange(0, len(dataset.pred_times), 10))
            ax.set_xticklabels(
                [ss for i, ss in enumerate(dataset.pred_times.str)
                 if i % 10 == 0],
                rotation=45
            )
            ax.legend()
        plt.subplots_adjust(bottom=0.2)
        fig.savefig(osp.join(root_dir, "R0.png"))

    elif args["model_type"] == "NetSEAIRQ":
        # 1. 整理params.csv
        new_index = ["Init number of %s" % i
                     for i in ["H", "R", "E", "A", "I", "Sq", "Eq"]]
        new_index += ["De", "Dq", "c", "beta", "q", "rho", "deltaI", "deltaQ"]
        new_index += ["gammaI", "gammaA", "gammaH", "theta", "nu", "phi"]
        new_index += ["c_k of %s" % i for i in dataset.regions]
        new_index += ["q_k of %s" % i for i in dataset.regions]

        new_values = list(model.y0for1)
        new_values += [
            getattr(model, i)
            for i in [
                "De", "Dq", "c", "beta", "q",
                "rho", "deltaI", "deltaQ",
                "gammaI", "gammaA", "gammaH",
                "theta", "nu", "phi"
            ]
        ]
        new_values += list(model.protect_args["c_k"])
        new_values += list(model.protect_args["q_k"])
        new_params = pd.Series(new_values, index=new_index)
        new_params.to_csv(osp.join(root_dir, "params_clear.csv"))

    # 3. 绘制 obj trace的log 尺度图
    opt_res = load(osp.join(root_dir, "opt_res.pkl"), "pkl")
    ObjTrace = opt_res["ObjTrace"]
    # import ipdb; ipdb.set_trace()
    # ObjTrace = np.log(ObjTrace)
    fig, ax = plt.subplots()
    ax.plot(ObjTrace)
    ax.set_yscale("log")
    fig.savefig(osp.join(root_dir, "LogObjTrace.png"))


if __name__ == "__main__":
    main()

import os.path as osp
import sys

import numpy as np
import pandas as pd

from utils import load


def main():
    # 读取数据
    root_dir = sys.argv[1]
    dat = load("./DATA/Provinces.pkl", "pkl")
    args = load(osp.join(root_dir, "args.json"), "json")

    # 1. 整理params.csv
    params = pd.read_csv(osp.join(root_dir, "params.csv"), index_col=0)["0"]
    R0 = params.iloc[0] * args["Di"]
    new_index = ["k of %s" % i for i in dat["regions"]]
    new_index = ["R0", "alpha_I", "y0 in 2019-12-01"] + new_index
    new_values = np.r_[
        R0, params.values[0],
        params.values[-1],
        params.values[1:-1]
    ]
    new_params = pd.Series(new_values, index=new_index)
    new_params.to_csv(osp.join(root_dir, "params_clear.csv"))


if __name__ == "__main__":
    main()

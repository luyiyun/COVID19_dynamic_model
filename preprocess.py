import os
import sys
import pickle
from dateutil.parser import parse

import numpy as np
import pandas as pd

import utils


def city_remove_end(cities):
    map_dict = {
        "黔西南布依族苗族": "黔西南",
        "克孜勒苏柯尔克孜": "克孜勒苏",
        "神农架林区": "神农架",
        "黔东南苗族侗族": "黔东南",
        "博尔塔拉蒙古": "博尔塔拉",
        "伊犁哈萨克": "伊犁",
        "巴音郭楞蒙古": "巴音郭楞"
    }

    clr_cities = [t.strip() for t in cities]
    clr_cities = [t[:-1] if t[-1] == "市" else t for t in clr_cities]
    clr_cities = [t[:-2] if t[-2:] == "地区" else t for t in clr_cities]
    clr_cities = [t[:-3] if t[-3:-1] == "自治" else t for t in clr_cities]
    clr_cities = [t[:-1] if t[-1] == "县" else t for t in clr_cities]
    clr_cities = [t[:-1] if t[-1] == "州" and len(t) > 2 else t
                  for t in clr_cities]
    clr_cities = [map_dict[t] if t in map_dict else t for t in clr_cities]
    clr_cities = [t[:2] if t[-1] == "族" else t for t in clr_cities]
    clr_cities = [t[:-1] if t[-1] == "盟" else t for t in clr_cities]
    return clr_cities


def province_remove_end(provs):
    map_dict = {
        "新疆维吾尔自治区": "新疆",
        "广西壮族自治区": "广西",
        "宁夏回族自治区": "宁夏"
    }
    clr_provs = [t.strip() for t in provs]
    clr_provs = [map_dict[t] if t in map_dict else t for t in provs]
    clr_provs = [t[:-1] if t[-1] == "省" else t for t in clr_provs]
    clr_provs = [t[:-3] if t[-3:-1] == "自治" else t for t in clr_provs]
    clr_provs = [t[:-1] if t[-1] == "市" else t for t in clr_provs]  # 直辖
    return clr_provs


def preprocess_cities_pmn():
    """
    处理各个城市间的人口流量关系，输出两个值：
    1. all_cities，是整理过后的每对城市、每个时间点上的流动人口比例，是一个df
    2. common_list，整理过后的城市列表
    """
    # 得到所有流量数据
    city_files = [f for f in os.listdir("./DATA/Original/Cities")
                  if f.endswith("csv")]
    city_names = [f[:-4] for f in city_files]
    print("一共有%d个城市。" % len(city_names))

    # 将所有城市的数据组合在一起
    all_cities = []
    for city_name in city_names:
        dat = pd.read_csv(os.path.join("./DATA/Original/Cities/",
                                       "%s.csv" % city_name),
                          index_col=0)
        dat = dat.loc[dat.source == city_name, :]
        dat.loc[:, "time"] = utils.clear_time(dat.loc[:, "time"].values)
        dat.loc[:, "value"] = utils.clear_value(dat.loc[:, "value"].values)
        all_cities.append(dat)
    all_cities = pd.concat(all_cities)
    print("最终合并得到的信息数: %d。" % all_cities.shape[0])

    # 清理城市数据，得到source和target都共有的数据
    all_cities.loc[:, "source"] = city_remove_end(
        all_cities.loc[:, "source"].values)
    all_cities.loc[:, "target"] = city_remove_end(
        all_cities.loc[:, "target"].values)
    source = set(all_cities.source)
    target = set(all_cities.target)
    common = source.intersection(target)
    source_lgl = all_cities.loc[:, "source"].isin(common)
    target_lgl = all_cities.loc[:, "target"].isin(common)
    all_cities = all_cities.loc[source_lgl & target_lgl, :]
    print("最终清理后的信息数: %d。" % all_cities.shape[0])

    # 根据时间将数据组合成matrix
    all_times = list(set(all_cities.loc[:, "time"].values))
    print("一共有%d个时间点。" % len(all_times))
    common_list = list(common)

    return all_cities, common_list


def preprocess_cities_epidemic():
    """ 对本次疫情的数据进行处理，返回处理过后的数据和记录的疫情开始的时间 """
    # 读取数据，并做初步的处理
    epidemic_dat = pd.read_csv("./DATA/Original/Cities_epidemic.csv")
    use_subset = (epidemic_dat.province != epidemic_dat.city) | \
        epidemic_dat.city.isin(["北京", "上海", "重庆", "天津"])
    epidemic_dat = epidemic_dat.loc[
        use_subset, ["city", "time", "cum_confirm"]]
    epidemic_dat = epidemic_dat.rename(columns={"city": "regions"})
    epidemic_dat = epidemic_dat.replace({"吉林市": "吉林"})
    # 将出现疫情最早的那一天看做t0，并将时间进行整理
    epidemic_dat.loc[:, "time"] = utils.clear_time(epidemic_dat.time.values)
    t0 = str(epidemic_dat.loc[:, "time"].min())  # 开始的时间
    epidemic_dat.loc[:, "time"] = epidemic_dat.loc[:, "time"].map(
        lambda x: utils.time_date2diff(x, t0))
    return epidemic_dat, t0


def preprocess_cities_population():
    """
    处理每个城市的人口数据，使用的数据是wiki百科中的2010年中国人口
    普查数据
    """
    # 读取数据
    population_dat = pd.read_csv("./DATA/Original/Cities_populaton.csv")
    population_dat.rename(
        columns={"cities": "regions", "people": "population"}, inplace=True
    )
    population_dat = population_dat.iloc[:-2, :]  # 最后面的澳门香港没有
    population_dat.loc[:, "regions"] = [r[:-1] if r[-1] == "市" else r[:-2]
                                        for r in population_dat.regions]
    population_dat.loc[:, "population"] = [float(s.replace(",", ""))
                                           for s in population_dat.population]
    return population_dat


def preprocess_cities():
    # 得到整理后的各个部分的数据
    print("开始preprocess_cities_pmn")
    pmn, regions = preprocess_cities_pmn()
    print("开始preprocess_cities_epidemic")
    epidemic, t0 = preprocess_cities_epidemic()
    print("开始preprocess_cities_population")
    population = preprocess_cities_population()

    # 将这些数据进行整合，取出三部分都有的地区
    set1, set2, set3 = (set(regions), set(epidemic.regions),
                        set(population.regions))
    common = list(set1.intersection(set2).intersection(set3))
    print("3个数据共同都有的城市数量是%d" % len(common))
    pmn = pmn[pmn.source.isin(common) & pmn.target.isin(common)]
    epidemic = epidemic[epidemic.regions.isin(common)]
    population = population[population.regions.isin(common)]

    # 把pmn的时间给整理成整数
    pmn.loc[:, "time"] = pmn.loc[:, "time"].map(
        lambda x: utils.time_date2diff(x, t0))
    pmn.loc[:, "value"] = pmn.loc[:, "value"] / 100

    # 将所有的城市替换成数字
    print("开始将城市替换成数字")
    cities_map = {city: i for i, city in enumerate(common)}
    pmn.replace(cities_map, inplace=True)
    epidemic.replace(cities_map, inplace=True)

    # 得到所有的时间点
    num_times = epidemic.time.max() + 1

    print("将dat转换成mat")
    population.set_index("regions", inplace=True)
    population_arr = population.loc[common, "population"].values
    epidemic_arr = utils.df_to_mat(epidemic, (num_times, len(common)),
                                   source="time", target="regions",
                                   values="cum_confirm")
    pmn_arrs = {}
    for d in set(pmn.time):
        pmn_arrs[d] = utils.df_to_mat(pmn[pmn.time == d],
                                      (len(common), len(common)))

    all_dat = {
        "t0": t0,
        "regions": common,
        "population": population_arr,
        "epidemic": epidemic_arr,
        "pmn": pmn_arrs,
        "num_times": num_times
    }
    with open("./DATA/City.pkl", "wb") as f:
        pickle.dump(all_dat, f)


def preprocess_provinces_pmn():
    """ preprocess_cities_pmn处理省份的翻版 """
    # 得到所有流量数据
    prov_files = [f for f in os.listdir("./DATA/Original/Provinces/")
                  if f.endswith("csv")]
    prov_names = [f[:-4] for f in prov_files]
    print("一共有%d个城市。" % len(prov_names))

    # 将所有数据组合在一起
    all_provs = []
    for prov_name in prov_names:
        dat = pd.read_csv(
            os.path.join("./DATA/Original/Provinces/", "%s.csv" % prov_name),
            index_col=0
        )
        dat.rename(columns={"target_province": "target"}, inplace=True)
        dat = dat.loc[dat.source == prov_name, :]
        dat.loc[:, "time"] = dat.time.map(lambda x: parse(x).toordinal())
        dat.loc[:, "value"] = dat.value.map(lambda x: float(x.strip()[:-1]))
        dat.loc[:, "value"] = dat.loc[:, "value"] / 100
        all_provs.append(dat)
    all_provs = pd.concat(all_provs)
    print("最终合并得到的信息数: %d。" % all_provs.shape[0])

    # 清理城市数据，得到source和target都共有的数据
    all_provs.loc[:, "source"] = province_remove_end(
        all_provs.loc[:, "source"].values)
    all_provs.loc[:, "target"] = province_remove_end(
        all_provs.loc[:, "target"].values)
    source = set(all_provs.source)
    target = set(all_provs.target)
    common = source.intersection(target)
    source_lgl = all_provs.loc[:, "source"].isin(common)
    target_lgl = all_provs.loc[:, "target"].isin(common)
    all_provs = all_provs.loc[source_lgl & target_lgl, :]
    print("最终清理后的信息数: %d。" % all_provs.shape[0])

    # 根据时间将数据组合成matrix
    all_times = list(set(all_provs.loc[:, "time"].values))
    print("一共有%d个时间点。" % len(all_times))
    common_list = list(common)
    return all_provs, common_list


def preprocess_provinces_epidemic2():
    # 读取数据，并做初步的处理
    epidemic_dat = pd.read_csv("./DATA/Original/Provinces_epidemic.csv")
    epidemic_dat = epidemic_dat.loc[:, ["province", "time", "cum_confirm"]]
    epidemic_dat = epidemic_dat.rename(columns={"province": "regions"})
    # 将出现疫情最早的那一天看做t0，并将时间进行整理
    epidemic_dat.loc[:, "time"] = utils.clear_time(epidemic_dat.time.values)
    t0 = str(epidemic_dat.loc[:, "time"].min())  # 开始的时间
    epidemic_dat.loc[:, "time"] = epidemic_dat.loc[:, "time"].map(
        lambda x: utils.time_date2diff(x, t0))
    return epidemic_dat, t0


def preprocess_provinces_epidemic():
    # 读取数据，并做初步的处理
    epidemic_dat = pd.read_csv("./DATA/Original/DXYArea0227.csv")
    epidemic_dat = epidemic_dat.iloc[:, [0, 1, 2, 3, -1]]
    epidemic_dat.rename(
        columns={"省": "region", "日期": "time"},
        inplace=True)
    epidemic_dat.drop_duplicates(["region", "time"], inplace=True)
    # 整理省的名称
    epidemic_dat.loc[:, "region"] = province_remove_end(epidemic_dat.region)
    # 整理values
    epidemic_dat["trueH"] = epidemic_dat.iloc[:, 1] \
        - epidemic_dat.iloc[:, 2] - epidemic_dat.iloc[:, 3]
    epidemic_dat.rename(columns={"省治愈": "trueR", "省死亡": "trueD"},
                        inplace=True)
    # 整理时间
    #   将出现疫情最早的那一天看做t0，并将时间进行整理，全部转换成ordinal格式
    epidemic_dat.loc[:, "time"] = epidemic_dat.time.map(
        lambda x: parse(x).toordinal())
    return epidemic_dat.loc[:, ["region", "time", "trueH", "trueR", "trueD"]]


def preprocess_provinces_population():
    # 读取数据
    population_dat = pd.read_csv("./DATA/Original/Provinces_population18.csv")
    population_dat.rename(
        columns={"provinces": "region", "people": "population"}, inplace=True
    )
    population_dat.loc[:, "population"] = population_dat.population * 10000
    return population_dat


def preprocesss_provinces_first_time():

    def parse_time(t):
        if isinstance(t, float):
            return np.nan
        t = "2020-0" + t.replace("月", "-").replace("日", "")
        return utils.time_str2ord(t)

    response_time = pd.read_csv(
        "./DATA/Original/ResponseTimeProvince.csv", header=None).iloc[:, :2]
    response_time.columns = ["region", "time"]
    response_time.loc[:, "time"] = response_time.time.map(parse_time)
    response_time.set_index("region", inplace=True)

    first_case = pd.read_csv("./DATA/Original/FirstProvince.csv", header=None)
    first_case.columns = ["region", "time"]
    first_case.loc[:, "time"] = first_case.time.map(parse_time)
    first_case.set_index("region", inplace=True)

    return response_time.time, first_case.time


def preprocess_provinces_outtrend():
    df = pd.read_csv("./DATA/Original/out_trend.csv")
    df = df.iloc[:, 1:]
    df.loc[:, "year19"] = df.year19 / 100
    df.loc[:, "year20"] = df.year20 / 100
    return df


def preprocess_provinces():
    # 得到整理后的各个部分的数据
    print("开始preprocess_provinces_pmn")
    pmn, regions = preprocess_provinces_pmn()
    print("开始preprocess_provinces_epidemic")
    epidemic = preprocess_provinces_epidemic()
    print("开始preprocess_provinces_population")
    population = preprocess_provinces_population()
    print("开始preprocess_provinces_first_time")
    response_time, first_case = preprocesss_provinces_first_time()
    print("开始preprocess_provinces_outtrend")
    out_trend = preprocess_provinces_outtrend()

    # 将这些数据进行整合，取出三部分都有的地区
    set1, set2, set3 = (set(regions), set(epidemic.region),
                        set(population.region))
    common = list(set1.intersection(set2).intersection(set3))
    common.remove("湖北")
    common.insert(0, "湖北")
    print("3个数据共同都有的省份数量是%d" % len(common))
    pmn = pmn[pmn.source.isin(common) & pmn.target.isin(common)]
    epidemic = epidemic[epidemic.region.isin(common)]
    population = population[population.region.isin(common)]

    # # 把pmn的时间给整理成整数
    # pmn.loc[:, "time"] = pmn.loc[:, "time"].map(
    #     lambda x: utils.time_date2diff(x, t0))
    # pmn.loc[:, "value"] = pmn.loc[:, "value"] / 100

    # 将所有的城市替换成数字
    print("开始将城市替换成数字")
    prov_map = {prov: i for i, prov in enumerate(common)}
    pmn.replace(prov_map, inplace=True)
    epidemic.replace(prov_map, inplace=True)
    out_trend.replace(prov_map, inplace=True)

    print("将dat转换成mat")
    # 整理population数据
    population.set_index("region", inplace=True)
    population_arr = population.loc[common, "population"].values
    # 整理epidemic数据
    epi_t0 = epidemic.time.min()
    epidemic.loc[:, "time"] = epidemic.time - epi_t0
    trueH_arr = utils.df_to_mat(
        epidemic, (epidemic.time.max()+1, len(common)),
        source="time", target="region", values="trueH"
    )
    trueR_arr = utils.df_to_mat(
        epidemic, (epidemic.time.max()+1, len(common)),
        source="time", target="region", values="trueR"
    )
    trueD_arr = utils.df_to_mat(
        epidemic, (epidemic.time.max()+1, len(common)),
        source="time", target="region", values="trueD"
    )

    # 整理pmn数据
    pmn_arrs = {}
    for d in set(pmn.time):
        mat = utils.df_to_mat(pmn[pmn.time == d], (len(common), len(common)))
        pmn_arrs[d] = mat  # 对于省份来时，这里没有遗漏的

    # 对out trend进行整理
    out_trend_t0 = out_trend.time.min()
    out_trend.loc[:, "time"] = out_trend.time - out_trend_t0
    out_trend20_df = out_trend[~out_trend.year20.isna()]
    out_trend20 = utils.df_to_mat(
        out_trend20_df,
        (out_trend20_df.time.max()+1, len(common)),
        source="time", target="region", values="year20"
    )
    out_trend19 = utils.df_to_mat(
        out_trend, (out_trend.time.max()+1, len(common)),
        source="time", target="region", values="year19"
    )

    all_dat = {
        "epidemic_t0": epi_t0,
        "out_trend_t0": out_trend_t0,
        "trueH": trueH_arr, "trueR": trueR_arr, "trueD": trueD_arr,
        "regions": common,
        "population": population_arr,
        "pmn": pmn_arrs,
        "response_time": response_time.loc[common].values,
        "first_case": first_case.loc[common].values,
        "out_trend20": out_trend20,
        "out_trend19": out_trend19
    }
    with open("./DATA/Provinces.pkl", "wb") as f:
        pickle.dump(all_dat, f)


if __name__ == "__main__":
    func_index = {
        "city": preprocess_cities,
        "province": preprocess_provinces,
    }

    args = sys.argv[1:]
    for arg in args:
        func = func_index[arg]
        print("")
        print("现在执行的是%s :%s" % (arg, func.__name__))
        func()

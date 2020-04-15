import time

import numpy as np
import geatpy as ea
from pathos.multiprocessing import Pool, cpu_count


class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, func, dim, lb, ub, njobs=0):
        self.func = func
        # 初始化name（函数名称，可以随意设置）
        name = 'MyProblem'
        # 初始化M（目标维数）
        M = 1
        # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        maxormins = [1]
        # 初始化Dim（决策变量维数）
        Dim = dim
        # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        varTypes = [0] * Dim
        # 决策变量下界
        # lb = [0] * Dim
        # 决策变量上界
        # ub = [1, 1, 3, 3]
        # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        lbin = [1] * Dim
        # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        ubin = [1] * Dim
        # 多进程
        self.njobs = njobs
        if njobs > 1:
            self.pool = Pool(njobs)
        elif njobs == -1:
            self.pool = Pool(int(cpu_count()))
        else:
            self.pool = None
        # 可视化
        self.count = 0
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(
            self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        x = pop.Phen  # 得到决策变量矩阵
        # 使用多进程进行
        if self.pool is not None:
            results = self.pool.map_async(self.func, list(x))
            results.wait()
            # 计算目标函数值，赋值给pop种群对象的ObjV属性
            results = np.array([results.get()]).T
        else:
            results = []
            for i in range(x.shape[0]):
                results.append(self.func(x[i, :]))
            results = np.stack(results, axis=0)
            results = np.expand_dims(results, 1)
        # 计算目标函数值，赋值给pop种群对象的ObjV属性
        pop.ObjV = results
        # 打印计数信息，便于进行查看
        self.count += 1
        print("第%d代完成" % self.count)


def geaty_func(
    func, dim, lb, ub, NIND=400, MAXGEN=25, fig_dir="",
    njobs=0, method="SEGA", n_populations=5
):
    """
    将整个遗传算法过程进行整合，编写成一整个函数，便于之后使用。当前此函数，只能处理参数是连续
    型的最优化问题。
    args：
        func: 用于最小化的函数。
        dim: 需要优化的params的个数。
        lb，ub：array或list，是params的上下限，这里默认都是闭区间。
        NIND: 种群规模
        MAXGEN: 最大进化代数
        fig_dir: 保存图片的地址，注意，需要在最后加/
        njobs: 0，1使用单核、否则使用多核，其中-1表示使用所有的核
        method: 使用的进化算法模板：
            SEGA：增强精英保留的遗传算法模板
            multiSEGA：增强精英保留的多种群协同遗传算法
            psySEGA：增强精英保留的多染色体遗传算法
            rand1lDE: 差分进化DE/rand/1/L算法
        n_populations: 如果是multiSEGA，此表示的是种群数量

    """

    """ 实例化问题对象 """
    problem = MyProblem(func, dim, lb, ub, njobs=njobs)  # 生成问题对象

    """ 种群设置 """
    # 创建区域描述器
    # (不同的方法使用不同的编码方式)
    if "DE" in method:
        Encoding = "RI"
    else:
        Encoding = "BG"
    # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
    if method == "multiSEGA":
        one_ind = NIND // n_populations
        res_ind = NIND % n_populations
        NINDS = [one_ind] * n_populations
        NINDS[-1] = NINDS[-1] + res_ind
        population = []
        for nind in NINDS:
            Field = ea.crtfld(
                Encoding, problem.varTypes, problem.ranges, problem.borders
            )
            population.append(ea.Population(Encoding, Field, nind))
    else:
        Field = ea.crtfld(
            Encoding, problem.varTypes, problem.ranges, problem.borders
        )
        if method == "psySEGA":
            raise NotImplementedError
            population = ea.PsyPopulation([Encoding], [Field], NIND)
        else:
            population = ea.Population(Encoding, Field, NIND)

    """ 算法参数设置 """
    if method == "SEGA":
        algorithm = ea.soea_SEGA_templet
    elif method == "multiSEGA":
        algorithm = ea.soea_multi_SEGA_templet
    elif method == "psySEGA":
        algorithm = ea.soea_psy_SEGA_templet
    elif method == "rand1lDE":
        algorithm = ea.soea_DE_rand_1_L_templet
    else:
        raise NotImplementedError
    # 实例化一个算法模板对象
    myAlgorithm = algorithm(problem, population)
    # 最大进化代数
    myAlgorithm.MAXGEN = MAXGEN
    # "进化停滞"判断阈值
    myAlgorithm.trappedValue = 1e-6
    # 进化停滞计数器最大上限值，如果连续maxTrappedCount代被判定进化陷入停滞，则终止进化
    myAlgorithm.maxTrappedCount = 5
    # 控制是否绘制图片
    myAlgorithm.drawing = 1
    # 控制绘图的路径（自己改的源码）
    myAlgorithm.drawing_file = fig_dir

    """ 调用算法模板进行种群进化 """
    [population, obj_trace, var_trace] = myAlgorithm.run()  # 执行算法模板
    # population.save()  # 把最后一代种群的信息保存到文件中

    """ 输出结果 """
    # 记录最优种群个体是在哪一代
    best_gen = np.argmin(problem.maxormins * obj_trace[:, 1])
    best_ObjV = obj_trace[best_gen, 1]
    print('最优的目标函数值为：%s' % (best_ObjV))
    # print('最优的控制变量值为：')
    # for i in range(var_trace.shape[1]):
    #     print(var_trace[best_gen, i])
    print('有效进化代数：%s' % (obj_trace.shape[0]))
    print('最优的一代是第 %s 代' % (best_gen + 1))
    print('评价次数：%s' % (myAlgorithm.evalsNum))
    print('时间已过 %s 秒' % (myAlgorithm.passTime))

    return {
        "BestObjv": best_ObjV, "BestParam": var_trace[best_gen, :],
        "AvailableEvoluion": obj_trace.shape[0],
        "EvaluationCount": myAlgorithm.evalsNum,
        "PassTime": myAlgorithm.passTime,
        "VarTrace": var_trace,
        "ObjTrace": obj_trace
    }

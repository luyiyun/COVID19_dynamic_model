import time

import numpy as np
import geatpy as ea
from pathos.multiprocessing import Pool, cpu_count


"""
该案例展示了一个简单的连续型决策变量最大化目标的单目标优化问题。
max f = x * np.sin(10 * np.pi * x) + 2.0
s.t.
-1 <= x <= 2
"""


class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, func, dim, lb, ub):
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
        self.pool = Pool(int(cpu_count()))
        # 可视化
        self.count = 0
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(
            self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        x = pop.Phen  # 得到决策变量矩阵
        results = self.pool.map_async(self.func, list(x))
        # for i in range(x.shape[0]):
        #     results.append(self.pool.apply_async(self.func, args=(x[i, :],)))
        # self.pool.close()
        # self.pool.join()
        # results = [res.get() for res in results]
        results.wait()
        # 计算目标函数值，赋值给pop种群对象的ObjV属性
        pop.ObjV = np.array([results.get()]).T
        self.count += 1
        print("第%d代完成" % self.count)


class MyAlgorithm(ea.soea_SEGA_templet):
    def __init__(self, problem, population, fig_dir):
        self.fig_dir = fig_dir
        super().__init__(problem, population)

    def finishing(self, population):
        # 处理进化记录器
        delIdx = np.where(np.isnan(self.obj_trace))[0]
        self.obj_trace = np.delete(self.obj_trace, delIdx, 0)
        self.var_trace = np.delete(self.var_trace, delIdx, 0)
        if self.obj_trace.shape[0] == 0:
            raise RuntimeError(
                'error: No feasible solution. (有效进化代数为0，没找到可行解。)')
        self.passTime += time.time() - self.timeSlot  # 更新用时记录
        # 绘图
        if self.drawing != 0:
            ea.trcplot(
                self.obj_trace,
                [['种群个体平均目标函数值', '种群最优个体目标函数值']],
                save_path=self.fig_dir,
                xlabels=[['Number of Generation']],
                ylabels=[['Value']], gridFlags=[[False]]
            )
        # 返回最后一代种群、进化记录器、变量记录器以及执行时间
        return [population, self.obj_trace, self.var_trace]


def geaty_func(
    func, dim, lb, ub, Encoding="BG", NIND=400, MAXGEN=25, fig_dir=""
):
    """
    Encoding 编码方式
    NIND 种群规模
    MAXGEN 最大进化代数
    """

    """ 实例化问题对象 """
    problem = MyProblem(func, dim, lb, ub)  # 生成问题对象

    """ 种群设置 """
    # 创建区域描述器
    Field = ea.crtfld(
        Encoding, problem.varTypes, problem.ranges, problem.borders
    )
    # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
    population = ea.Population(Encoding, Field, NIND)

    """ 算法参数设置 """
    # 实例化一个算法模板对象
    myAlgorithm = MyAlgorithm(problem, population, fig_dir)
    # 最大进化代数
    myAlgorithm.MAXGEN = MAXGEN
    # 控制是否绘制图片
    myAlgorithm.drawing = 1

    """ 调用算法模板进行种群进化 """
    [population, obj_trace, var_trace] = myAlgorithm.run()  # 执行算法模板
    # population.save()  # 把最后一代种群的信息保存到文件中

    """ 输出结果 """
    # 记录最优种群个体是在哪一代
    best_gen = np.argmin(problem.maxormins * obj_trace[:, 1])
    best_ObjV = obj_trace[best_gen, 1]
    print('最优的目标函数值为：%s' % (best_ObjV))
    print('最优的控制变量值为：')
    for i in range(var_trace.shape[1]):
        print(var_trace[best_gen, i])
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

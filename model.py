import inspect
from copy import deepcopy

import numpy as np
from scipy import integrate
from scipy.optimize import dual_annealing

import utils
from geatyAlg import geaty_func


class InfectiousBase:
    """
    这是一个用于构建传染病动力学模型的基类，其记录了我们需要实现的各种方法
    子类主要需要实现的方法有：
        __init__
        __call__
        score (需要用到predict)
        R
    还需要进一步封装的方法有：
        predict
    如果需要进行训练，则还需要实现：
        _set_args和_get_fit_x0
    """

    def __init__(self, *args, **kwargs):
        """
        接受的是子类初始化时的参数，会将其全部设为实例属性，并保存到kwargs属性中，便于之后保存
        和读取时使用
        """
        self.rtol, self.atol = 1e-8, 1e-8  # 1e-5会出现问题
        # 将子类所有的参数都记录到属性kwargs中
        sig = inspect.signature(self.__init__)
        arg_names = list(sig.parameters.keys())
        for argname, arg in zip(arg_names, args):
            kwargs[argname] = arg
        self.kwargs = kwargs
        # 将子类的参数都变成子类实例的属性
        for k, v in self.kwargs.items():
            setattr(self, k, v)

    def __call__(self, t, vars):
        """
        实现微分方程标准形式的右侧部分，用于输入到solve_ivp中进行求解

        Arguments:
            t {float} -- 时间
            vars {ndarray} -- 当前时间点上的值，shape等于方程的数量

        Raises:
            NotImplementedError

        Returns:
            ndarray -- 当前各个值的导数
        """
        raise NotImplementedError

    def predict(self, times):
        """
        预测，这里得到的是所有预测值，一般来说，还需要在子类中对其进行进一步的封装和分类

        Arguments:
            times {ndarray} -- 想要计算的时间点

        Returns:
            ndarray -- shape=(n_times, n_equ)
        """
        t_span = (0, times.max())
        results = integrate.solve_ivp(
            self, t_span=t_span, y0=self.y0,
            t_eval=times, rtol=self.rtol, atol=self.atol,
            # method="RK23"
        )
        return results.y.T

    def score(self, times, trues, mask=None):
        """
        评价指标

        Arguments:
            times {ndarray} -- 时间点
            trues {ndarray} -- 对应时间点上的真实值

        Keyword Arguments:
            mask {ndarray} -- 掩模，其=0或=False的地方会覆盖掉times和trues，被覆盖掉的
                值不会参与计算 (default: {None})

        Raises:
            NotImplementedError: [description]

        Returns:
            float -- 评价得分，越小越好
        """
        raise NotImplementedError

    @property
    def R(self):
        """ 估计基本再生数 """
        raise NotImplementedError

    def save(self, filename):
        """
        保存

        Arguments:
            filename {str} -- 保存的地址
        """
        utils.save(self.kwargs, filename, "pkl")

    @classmethod
    def load(cls, filename):
        """
        载入模型

        Arguments:
            filename {str} -- 载入保存的模型或模型列表

        Returns:
            Infectious or list -- 重新实例化的模型或模型列表
        """
        configs = utils.load(filename, "pkl")
        if isinstance(configs, dict):
            return cls(**configs)
        elif isinstance(configs, list):
            return [cls(**config) for config in configs]
        else:
            raise ValueError

    @property
    def fit_params_info(self):
        """
        1. 这里记录我们需要更新的参数的信息，如果想要变换我们更新的参数，就在这里更改，来方便
        程序的实验。
        2. 这里使用OrderDict进行记录，键为其对应的属性名，而值是这个参数的(维度, 下限，上限)
        3. 如果key使用A-B的格式，则这里表示的是self.A["B"]的值
        4. 如果key中在最后有[n1:n2]的字样，则表示当前要将params赋值到该参数的n1:n2切片上，
            当然，也可以是[n1]表示单个值的定位
        5. 可以选择性的在value最后再跟一个列表，其中每个元素对应的是当前参数的解释，可以在打印
            信息的时候使用

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError

    def fit_params_range(self):
        """
        得到所有需要拟合参数的维度、拟合范围

        Returns:
            (int, ndarray, ndarray) -- 参数向量维度、拟合下限，拟合上限（包含这个界限）
        """
        num_params, lb, ub = 0, [], []
        for vs in self.fit_params_info.values():
            n, l, u = vs[:3]
            num_params += n
            lb.extend([l] * n)
            ub.extend([u] * n)
        return num_params, np.array(lb), np.array(ub)

    def set_params(self, params):
        """
        将params向量中储存的拟合参数信息整合到模型中

        Arguments:
            params {ndarray} -- shape = 所有参数的总和
        """
        i = 0
        for k, v in self.fit_params_info.items():
            key1, key2, ind = utils.parser_key(k)
            if key2 is not None:
                if ind is not None:
                    self.kwargs[key1][key2][ind] = params[i:(i+v[0])]
                else:
                    self.kwargs[key1][key2] = params[i:(i+v[0])]
            else:
                if ind is not None:
                    self.kwargs[key1][ind] = params[i:(i+v[0])]
                else:
                    self.kwargs[key1] = params[i:(i+v[0])]
            i += v[0]
        self.__init__(**self.kwargs)


def score_func(params, model, score_kwargs):
    """
    得到对应参数的得分

    Arguments:
        params {ndarray} -- 当前参数组成的ndarray，shape=(参数量,)
        model {Infectious实例} -- 需要拟合的模型
        score_kwargs {dict} -- model.score方法需要的一些参数

    Returns:
        float -- 评价得分
    """
    model_copy = deepcopy(model)  # copy一下，防止在多进程时造成数据的错误
    model_copy.set_params(params)
    return model_copy.score(**score_kwargs)


def find_best(func, dim, lb, ub, method="SEGA", **kwargs):
    """
    寻找当前函数的最小值对应的参数

    Arguments:
        func {callable} -- 单输入函数，返回float
        dim {int} -- 拟合参数数量
        lb {ndarray} -- 各个参数的下限
        ub {ndarray} -- 各个参数的上限

    Keyword Arguments:
        method {str} -- [description] (default: {"SEGA"})
        kwargs -- 各个最优化方法需要的其他参数

    Raises:
        NotImplementedError: [description]

    Returns:
        dict -- opt_res，不同的方法内容不同，但都有一个BestParams，其中的内容是一个
            ndarray，表示找到的最优参数
    """
    if method == "annealing":
        opt_res_1 = dual_annealing(func, np.stack([lb, ub], axis=1), **kwargs)
        opt_res = {"all": opt_res_1, "BestParam": opt_res_1.x}
    else:
        opt_res = geaty_func(
            func, dim=dim, lb=lb, ub=ub, method=method, **kwargs)
    return opt_res


def save_models(models, filename):
    """
    保存模型组成的list

    Arguments:
        models {list} -- 里面每个元素是一个Infectious模型
        filename {str} -- 保存的文件名
    """
    kwarg_list = [model.kwargs for model in models]
    utils.save(kwarg_list, filename)

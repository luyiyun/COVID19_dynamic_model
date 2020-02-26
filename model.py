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
        注意，实例化子类的时候，记得把所有的参数都放到super().__init__()中，这样可以自动
        记忆所有的参数，便于之后保存和读取的时候使用
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
        这里实现的是微分方程组标准形式的右侧部分。
        用于输入到`solve_ivp`中进行求解
        """
        raise NotImplementedError

    def score(self, times, trues, mask=None):
        """
        输出评价指标，参数times是时间点，trues是真实的每个时间点上的值，
        ndarray
        """
        raise NotImplementedError

    def predict(self, times):
        """
        输入想要预测的时间点，得到其预测值
        需要是ndarray
        返回的是ndarray, dim = len(times) * len(vars)
        注意，这里得到的是所有预测值，一般来说，还需要在子类中对其进行进一步的封装
        """
        t_span = (0, times.max())
        results = integrate.solve_ivp(
            self, t_span=t_span, y0=self.y0,
            t_eval=times, rtol=self.rtol, atol=self.atol,
            # method="RK23"
        )
        return results.y.T

    @property
    def R(self):
        """
        估计基本再生数
        """
        raise NotImplementedError

    def save(self, filename):
        """ 保存参数, 使用pkl可以保存ndarray """
        utils.save(self.kwargs, filename, "pkl")

    @classmethod
    def load(cls, filename):
        configs = utils.load(filename, "pkl")
        return cls(**configs)

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
        """
        raise NotImplementedError

    def fit_params_range(self):
        """ 利用fit_params_info记录的参数信息，返回所有参数的维度、拟合范围 """
        num_params, lb, ub = 0, [], []
        for vs in self.fit_params_info.values():
            n, l, u = vs[:3]
            num_params += n
            lb.extend([l] * n)
            ub.extend([u] * n)
        return num_params, np.array(lb), np.array(ub)

    def set_params(self, params):
        """
        利用的是fit_params_info的信息
        将当前的参数设置到模型中，注意，需要使用运行init方法，这样才能更新一些参数，
        比如y0、gammafunc等
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
    # try:
    # 有些不好的参数，可能会导致微分方程不收敛，则会保存，这时我们认为这些参数是坏的，
    # 返回一个大值表示其loss
    model_copy = deepcopy(model)
    model_copy.set_params(params)
    return model_copy.score(**score_kwargs)
    # except ValueError:
    # return np.inf


def find_best(func, dim, lb, ub, method="geatpy", **kwargs):
    """
    kwargs: 不同拟合方法所需要的额外参数
    """
    if method == "geatpy":
        opt_res = geaty_func(
            func, dim=dim, lb=lb, ub=ub,
            **kwargs,
            # Encoding="BG", NIND=400, MAXGEN=25,
            # fig_dir=save_dir+"/", njobs=1
        )
    elif method == "annealing":
        opt_res_1 = dual_annealing(
            func, np.stack([lb, ub], axis=1),
            **kwargs,
            # maxiter=1000,
            # callback=utils.callback
        )
        opt_res = {"all": opt_res_1, "BestParam": opt_res_1.x}
    else:
        raise NotImplementedError
    return opt_res

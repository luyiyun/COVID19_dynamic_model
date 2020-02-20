import inspect

from scipy import integrate, optimize
from sko.DE import DE
from skopt import gp_minimize

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
        self.rtol, self.atol = 1e-8, 1e-8
        # 将子类所有的参数都记录到属性kwargs中
        sig = inspect.signature(self.__init__)
        arg_names = list(sig.parameters.keys())
        for argname, arg in zip(arg_names, args):
            kwargs[argname] = arg
        self.kwargs = kwargs
        # 将子类的参数都变成子类实例的属性
        for k, v in self.kwargs.items():
            setattr(self, k, v)
        self._fit_count = 0

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
        # import ipdb; ipdb.set_trace()
        # t = np.r_[0, times]
        t_span = (0, times.max())
        results = integrate.solve_ivp(
            self, t_span=t_span, y0=self.y0,
            t_eval=times,
            # t_eval=times,
            rtol=self.rtol, atol=self.atol)
        # results = integrate.odeint(
        #     self, self.y0, t, tfirst=True, rtol=self.rtol, atol=self.atol,
        #     full_output=True
        # )
        self._fit_count += 1
        print("fit counts: %d" % self._fit_count)
        return results.y.T
        # return results[0][1:, :]

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

    def fit(self, true_times, true_values, mask=None, *args, **kwargs):
        # 创建一个函数，其以参数为输入，score为输出
        def score_func(params):
            self._set_args(params, *args, **kwargs)
            return self.score(true_times, true_values, mask=mask)

        # 得到对应于不同方法的x0或dimension
        x0 = self._get_fit_x0()

        # 根据不同的方法进行拟合参数
        if self.fit_method == "scipy-NM":
            self.opt_res = optimize.minimize(
                score_func, x0,
                method="Nelder-Mead", callback=self._NM_callback,
                options={"disp": False, "adaptive": True})
            best_x = self.opt_res.x
        elif self.fit_method == "scikit-optimize":
            self.opt_res = gp_minimize(
                score_func, dimensions=x0, verbose=True, n_calls=100)
            best_x = self.opt_res.x
        elif self.fit_method == "scikit-opt":
            self.opt_res = DE(score_func, n_dim=x0[0],
                              size_pop=500, max_iter=1000,
                              lb=x0[1], ub=x0[2])
            best_x, _ = self.opt_res.run()
        elif self.fit_method == "geaty":
            self.opt_res = geaty_func(
                score_func, x0[0], x0[1], x0[2],
                NIND=400, MAXGEN=25
            )
            best_x = self.opt_res["BestParam"]
        self._set_args(best_x, *args, **kwargs)
        return self

    def _set_args(self, params, *args, **kwargs):
        raise NotImplementedError

    def _get_fit_x0(self):
        raise NotImplementedError

    def _NM_callback(self, Xi):
        if self._fit_count == 0:
            print("%4s   %9s   %9s   %9s" %
                  ("Iter", "alpha_E", "alpha_I", "k"))
        print("%4d   %3.6f   %3.6f   %3.6f" %
              (self._fit_count, Xi[0], Xi[1], Xi[2]))
        self._fit_count += 1

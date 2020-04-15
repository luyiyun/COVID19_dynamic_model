import torch
import pyro
import pyro.distributions as dist
from pyro.distributions import constraints
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO, NUTS, MCMC, Predictive
from pyro.infer.autoguide import AutoMultivariateNormal, init_to_mean
import pymc3 as pm
from tqdm import trange


class BayesInferece:

    def __init__(self, pred_func, dim, lb, ub):
        self.pred_func = pred_func
        self.fit_dim = dim
        self.lb, self.ub = torch.tensor(lb).float(), torch.tensor(ub).float()

    def model(self, data):
        """
        这个data是每个省份每一天的真实值，是2D数据
        """
        params_sample = []
        # for i in pyro.plate("param_plate", self.fit_dim):
        for i in range(self.fit_dim):
            param_one = pyro.sample(
                "fit_params_%d" % i, dist.Uniform(self.lb[i], self.ub[i])
            )
            params_sample.append(param_one.detach().item())
        params_sample = np.array(params_sample)
        # import ipdb; ipdb.set_trace()
        pred_mean = self.pred_func(params_sample)  # 使用当前的参数进行预测
        pred_mean = torch.tensor(pred_mean)
        pyro.sample("obs", dist.Poisson(pred_mean), obs=data)
        # plate1 = pyro.plate("obs_plate_1", data.shape[-1])
        # plate2 = pyro.plate("obs_plate_2", data.shape[-2])
        # for i in plate1:
        #     for j in plate2:
        #         pyro.sample("obs_i%d_j%d" % (i, j),
        #                     dist.Poisson(pred_mean), obs=data[j, i])

    def guide(self, data):
        mean = pyro.param(
            "mean", torch.rand(self.fit_dim),
            # constraint=constraints.positive
        )
        std = pyro.param(
            "std", torch.rand(self.fit_dim),
            constraint=constraints.positive
        )
        with pyro.plate("guide_plate", self.fit_dim):
            pyro.sample("fit_params", dist.Normal(mean, std))

    def multi_norm_guide(self):
        return AutoMultivariateNormal(self.model, init_loc_fn=init_to_mean)

    def inference_svi(self, data, steps=3000, lr=0.01):
        self.inference_method = "svi"
        pyro.clear_param_store()
        self.optimizer = Adam({"lr": lr, "betas": (0.90, 0.999)})
        self.svi = SVI(
            self.model, self.multi_norm_guide(),
            self.optimizer, loss=Trace_ELBO()
        )
        self.history = {"losses": []}
        data = torch.tensor(data).float()
        bar = trange(steps)
        for i in bar:
            loss = self.svi.step(data)
            if (i+1) % 100 == 1:
                bar.write("Now step %d completed, loss is %.4f" % (i, loss))
            self.history["losses"].append(loss)

    def inference_mcmc(
        self, data, num_samples=3000, warmup_steps=2000, num_chains=4
    ):
        self.inference_method = "mcmc"
        data = torch.tensor(data).float()
        self.nuts_kernel = NUTS(self.model, adapt_step_size=True)
        self.mcmc = MCMC(
            self.nuts_kernel, num_samples=num_samples,
            warmup_steps=warmup_steps, num_chains=num_chains
        )
        self.mcmc.run(data)

    def save(self, path):
        pass

    def fit_params_estimate(self, data):
        if self.inference_method == "svi":
            data = torch.tensor(data).float()
            predictive = Predictive(
                self.model, guide=self.multi_norm_guide(), num_samples=1000)
            svi_samples = {
                k: v.reshape(1000).detach().numpy()
                for k, v in predictive(data).items()
                if not k.startswith("obs")
            }
            return {
                "mean": np.array([v.mean() for v in svi_samples.values()]),
                "std": np.array([v.std() for v in svi_samples.values()])
            }

        elif self.inference_method == "mcmc":
            return {
                "mean": self.mcmc.get_samples()["fit_params"].mean(0)
            }
        else:
            raise NotImplementedError


class BayesInferece_PyMC:
    def __init__(self, pred_func, dim, lb, ub):
        self.pred_func = pred_func
        self.fit_dim = dim
        self.lb, self.ub = lb, ub

    def inference_mcmc(
        self, data, num_samples=3000, warmup_steps=2000, num_chains=4
    ):
        """
        这个data是每个省份每一天的真实值，是2D数据
        """
        with pm.Model() as model:
            fit_params = []
            for i in range(self.fit_dim):
                fit_param = pm.Uniform(
                    "fit_params_%d" % i, lower=self.lb[i], upper=self.ub[i])
                fit_params.append(fit_param)
            pred_mean = self.pred_func(np.array(fit_params))
            for i in range(pred_mean.shape[0]):
                for j in range(pred_mean.shape[1]):
                    likelihood = pm.Poisson(
                        "obs_%d_%d" % (i, j),
                        mu=pred_mean[i, j], observed=data[i, j]
                    )
            posterior = pm.sample(num_samples, cores=num_chains)
            posterior_pred = pm.sample_posterior_predictive(posterior)

        return posterior, posterior_pred

    def save(self, path):
        pass

    def fit_params_estimate(self, data):
        pass


if __name__ == "__main__":
    import numpy as np

    def exam_func(x):
        return x.reshape(5, 2)
    true_mean = np.random.rand(10)
    # linear_trans = np.random.rand(10, 31)
    # after_trans = np.matmul(linear_trans, true_mean).reshape(5, 2)
    after_trans = exam_func(true_mean)
    data = torch.distributions.Poisson(torch.tensor(after_trans)).sample()
    data = data.numpy()
    # print(data)

    # estimator = BayesInferece(
    #     # lambda x: np.matmul(linear_trans, x).reshape(5, 2),
    #     exam_func,
    #     10, np.zeros(10), np.ones(10)
    # )
    estimator = BayesInferece_PyMC(
        exam_func,
        10, np.zeros(10), np.ones(10)
    )
    # estimator.inference_svi(data, steps=5000, lr=0.001)
    post, post_pred = estimator.inference_mcmc(data, 1000, 200)
    import ipdb; ipdb.set_trace()

    print("true_mean", true_mean)
    print("estimate_mean", estimator.fit_params_estimate(data))
    import ipdb; ipdb.set_trace()

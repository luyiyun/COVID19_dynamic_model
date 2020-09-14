from math import exp, ceil

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import numpy as np
import scipy.integrate as integrate
from tqdm import tqdm
import matplotlib.pyplot as plt


def R2I(R0, De, Di):
    return R0 / (2 / (1 / De + 1 / Di))


def protect(t, k):
    return exp(-k * t)


def Differential(t, y, R0, De, Di, k=0):
    S, E, I, _ = y[0], y[1], y[2], [3]
    N = y.sum()
    alpha = R2I(R0, De, Di) * protect(t, k)
    beta = 1 / De
    gamma = 1 / Di
    E_ = alpha * S * I / N - beta * E
    I_ = beta * E - gamma * I
    R_ = gamma * I
    S_ = -alpha * S * I / N
    return np.r_[S_, E_, I_, R_]


def R0_pred(R0, De, Di, k=0, y0=1, N=100000, interval=(0, 100)):
    ode_res = integrate.solve_ivp(
        lambda t, y: Differential(t, y, R0, De, Di, k),
        [0, max(*interval)],
        np.array([N-y0, 0, y0, 0]),
        method='RK45',
        t_eval=np.arange(*interval),
    )
    return ode_res.y[2, :]


class ODEdata(data.IterableDataset):
    def __init__(self, num_samples, lb, ub, gen_func, random_seed=0):
        super().__init__()
        self.num_samples = num_samples
        self.lb, self.ub = lb, ub
        self.gen_func = gen_func
        self.nparams = len(self.lb)
        self.seed = random_seed

    def __iter__(self):
        params = np.random.uniform(self.lb, self.ub,
                                   size=(self.num_samples, self.nparams))
        for i in range(self.num_samples):
            yield (params[i], self.gen_func(params[i]))

    def __len__(self):
        return self.num_samples

    def set_seed(self, seed=None):
        if seed is not None:
            self.seed = seed
        np.random.seed(self.seed)


def worker_init_fn(worker_id):
    worker_info = data.get_worker_info()
    dataset = worker_info.dataset
    sample_i = int(ceil(dataset.num_samples / float(worker_info.num_workers)))
    if worker_id == (worker_info.num_workers - 1):
        sample_i = int(dataset.num_samples - worker_id * sample_i)
    dataset.num_samples = sample_i
    dataset.set_seed(worker_id+dataset.seed)


class AuxiliaryNN(nn.Module):
    def __init__(self, in_dim, ou_dim, mlp_hiddens, lstm_hidden, num_lstm):
        super().__init__()
        self.in_dim, self.ou_dim = in_dim, ou_dim
        self.lstm_hidden = lstm_hidden
        lin_nns = []
        for i, j in zip([in_dim] + list(mlp_hiddens)[:-1], list(mlp_hiddens)):
            lin_nns.append(nn.Linear(i, j))
            lin_nns.append(nn.BatchNorm1d(j))
            lin_nns.append(nn.ReLU())
        self.lin_nns = nn.Sequential(*lin_nns)
        # self.lstm = nn.LSTMCell(mlp_hiddens[-1], lstm_hidden)
        # self.lstm_inp = nn.Linear(1, mlp_hiddens[-1])
        # self.lstm_out = nn.Linear(lstm_hidden, ou_dim)
        self.lstm = nn.LSTM(
            input_size=1+mlp_hiddens[-1], hidden_size=lstm_hidden,
            num_layers=num_lstm, batch_first=True
        )
        self.output = nn.Linear(lstm_hidden, ou_dim)

    def forward(self, x, time_inds):
        """
        Arguments:
            x: shape=(batch, nparams)
            time_inds: shape=(ntimes,)
        Return:
            output: shape=(batch, ntimes, ou_dim)
        """
        time_inds = time_inds.unsqueeze(0).expand(x.size(0), -1).unsqueeze(-1)
        feature = self.lin_nns(x)
        lstm_inpt = feature.unsqueeze(1).expand(-1, time_inds.size(1), -1)
        lstm_inpt = torch.cat([lstm_inpt, time_inds], axis=-1)

        output, _ = self.lstm(lstm_inpt)
        reshape_output = output.reshape(-1, self.lstm_hidden)
        reshape_output = self.output(reshape_output).exp()
        # h, c = feature, feature
        # output = []
        # for i in range(time_inds.size(1)):
        #     inpt = self.lstm_inp(time_inds[:, i, :])
        #     h, c = self.lstm(inpt, (h, c))
        #     oupt = self.lstm_out(h).exp()
        #     output.append(oupt)
        # return torch.stack(output, dim=1)
        return reshape_output.reshape(
            output.size(0), output.size(1), self.ou_dim
        )


class AuxiliaryNN2(nn.Module):
    def __init__(self,):
        pass

    def forward(self, x):
        pass


if __name__ == "__main__":
    # setup
    num_samples = 100000
    batch_size = 256
    num_workers = 4
    lr = 0.001
    ntimes = 5

    # 创建数据集
    dat = ODEdata(
        num_samples, [0, 0, 0], [5, 10, 20],
        lambda x: R0_pred(x[0], x[1], x[2], interval=(0, ntimes))
        )
    train_data = data.DataLoader(
        dat, batch_size=batch_size, num_workers=5,
        worker_init_fn=worker_init_fn
    )

    # 实例化model、loss_fn、optim等
    model = AuxiliaryNN(3, 1, [5], 10, 5).cuda()
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # train
    history = {"train": []}
    for i in range(5):
        with tqdm(total=ceil(num_samples/batch_size)) as t:
            t.set_description("Epoch: %d" % i)
            for inpt, oupt in train_data:
                inpt, oupt = inpt.cuda().float(), oupt.cuda().float()
                pred = model(inpt, torch.arange(ntimes).cuda().float())
                loss = loss_fn(pred, oupt.unsqueeze(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.detach().cpu().item()
                t.update()
                t.set_postfix(train_loss=loss_item)
                history["train"].append(loss_item)

    # 绘制loss
    plt.plot(history["train"])
    plt.savefig("./VNNexam.png")
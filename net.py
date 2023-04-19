import torch
import torch.nn as nn


class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)


class ClassifierNetwork(nn.Module):
    def __init__(self, input_dim, n_class):
        super().__init__()
        if input_dim == 124:
            self.clf = nn.Sequential(
                nn.BatchNorm1d(input_dim, affine=False),
                nn.Linear(input_dim, 128),
                nn.ReLU(), nn.Dropout(0.5),
                nn.BatchNorm1d(128, affine=False),
                nn.Linear(128, n_class),
                nn.Softmax(dim=-1)
            )
        elif input_dim == 8192:
            self.clf = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(128, n_class),
                nn.Softmax(dim=-1)
            )
        elif input_dim == 512:
            self.clf = nn.Sequential(
                nn.Linear(input_dim, 4096),
                nn.ReLU(True), nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(True), nn.Dropout(0.5),
                nn.Linear(4096, n_class),
                nn.Softmax(dim=-1)
            )

    def forward(self, x):
        o = self.clf(x)
        return o


class ReliabilityNetwork(nn.Module):
    def __init__(self, input_dim, n_class):
        super().__init__()
        self.pred1 = nn.Linear(input_dim, 128)
        self.pred2 = nn.Linear(128, n_class)
        self.pred3 = nn.Linear(n_class, 1)
        self.n_class = n_class

    def forward(self, x, w, q):
        x = torch.sigmoid(self.pred1(x))
        x = torch.sigmoid(self.pred2(x))
        w = torch.sigmoid(w)
        p = (w[None, :, :] - x[:, None, :]) * q[:, None, :]
        p = torch.sigmoid(self.pred3(p))
        return p

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.pred3.apply(clipper)


class AnnotateNetwork(nn.Module):
    def __init__(self, n_worker, n_class, input_dim):
        super().__init__()
        self.expertise = nn.Parameter(torch.empty(n_worker, n_class), requires_grad=True)
        nn.init.xavier_normal_(self.expertise)
        self.confusions = nn.Parameter(torch.stack([torch.eye(n_class, n_class) for _ in range(n_worker)]),
                                       requires_grad=True)
        self.prob = ReliabilityNetwork(input_dim, n_class)
        self.clf = ClassifierNetwork(input_dim, n_class)

    def forward(self, x, mode):
        cls_out = self.clf(x)
        crowd_out = None
        p = self.prob(x, self.expertise, cls_out)
        if mode == 'train':
            crowd_out = torch.einsum('ik,jkl->ijl', cls_out, self.confusions)
        return cls_out, crowd_out, p.squeeze()

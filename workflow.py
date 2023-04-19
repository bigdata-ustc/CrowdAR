import warnings

import torch
import torch.nn as nn
from sklearn import metrics

warnings.filterwarnings("ignore")


def soft_label(anno, p, n_class):
    _eye, _one = torch.eye(n_class).cuda(), torch.ones(n_class).cuda() / n_class
    onehot_anno = torch.stack([_eye[k] for k in anno.view(-1)])
    soft_anno = p[:, None] * onehot_anno + (1 - p[:, None]) * _one[None, :]
    return soft_anno


def multi_loss(y_pred, y_true, p, n_class, loss_fn=nn.CrossEntropyLoss(reduction='mean').cuda()):
    mask = y_true != -1
    soft_y_true = soft_label(y_true[mask], p[mask], n_class)
    loss = loss_fn(y_pred[mask], soft_y_true)
    return loss


def talking_loss(cls_out, p, anno):
    idx = torch.where(anno != -1)
    p_true = torch.where(torch.argmax(cls_out, dim=1)[:, None] == anno, 1, 0)
    p0 = torch.ones(p.size()).cuda() - p
    o = torch.cat((p0, p), 1)
    loss = nn.NLLLoss()(torch.log(o[idx]), p_true[idx])
    return loss


def test(model, test_loader):
    model.eval()
    pred, truth = [], []
    for _, x, y in test_loader:
        cls_out, _, _ = model(x.cuda(), mode='test')
        truth.extend(y.numpy().tolist())
        pred.extend(torch.argmax(cls_out, dim=1).cpu().detach().numpy().tolist())
    acc = metrics.accuracy_score(truth, pred)
    f1_score = metrics.f1_score(truth, pred, average='macro')
    precision = metrics.precision_score(truth, pred, average=None)
    return acc, f1_score, precision


def train(model, optimizer, train_loader, n_class, term=0.1):
    model.train()
    train_loss = []
    pred, truth = [], []
    for _, x, anno, y in train_loader:
        x, anno = x.cuda(), anno.cuda()
        cls_out, crowd_out, p = model(x, mode='train')
        loss = (1 - term) * multi_loss(crowd_out, anno, p, n_class) + term * talking_loss(cls_out, p, anno)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.prob.apply_clipper()
        train_loss.append(loss.item())
        pred.extend(torch.argmax(cls_out, dim=1).cpu().detach().numpy().tolist())
        truth.extend(y.tolist())
    acc = metrics.accuracy_score(truth, pred)
    return acc

import argparse
import logging
import random
from copy import deepcopy

import torch.optim as optim
from torch.utils.data import DataLoader

from net import *
from utils import *
from workflow import *


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def inference_method(e, lr, term):
    model = AnnotateNetwork(n_worker=num_workers, n_class=num_classes, input_dim=input_size).cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_test_acc, best_model, best_e = 0, None, 0
    for _e in range(e):
        acc = train(model, optimizer, train_loader, num_classes, term)
        test_acc, f1_score, precision = test(model, test_loader)
        if best_test_acc < test_acc:
            best_test_acc = test_acc
            best_model = deepcopy(model)
            best_e = _e
    msg = f'e {best_e} test_acc {best_test_acc:.5f}'
    return msg, best_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters")
    parser.add_argument('--d', type=str, default='labelme', help="dataset")
    parser.add_argument('--bs', type=int, default=256, help="batch size")
    parser.add_argument('--lr', type=float, default=0.005, help="learning rate")
    parser.add_argument('--t', type=float, default=0.1, help="term for adjusting two losses' weight")
    parser.add_argument('--e', type=int, default=400, help="epoch")
    parser.add_argument("--s", type=int, default=42, help="random seed")
    args = parser.parse_args()
    setup_seed(args.s)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO,
                        filename=f'result.log',
                        filemode='a')
    logger = logging.getLogger()
    train_dataset = CrowdDataset(mode='train', dataset=args.d)
    test_dataset = CrowdDataset(mode='test', dataset=args.d)
    test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
    num_tasks, num_workers = train_dataset.anno.shape[0], train_dataset.anno.shape[1]
    num_classes, input_size = train_dataset.num_classes, train_dataset.input_size
    msg, model = inference_method(e=args.e, lr=args.lr, term=args.t)
    logger.info(f'dataset {args.d} seed {args.s} bs {args.bs} term {args.t} lr {args.lr} {msg}')

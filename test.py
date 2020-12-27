import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import matplotlib.pyplot as plt
import random
import skimage.io as io
from System_parameter import args


def spiral_kxky(filename, ledNum):
    kxky = [[], []]
    with open(filename, 'r') as file:
        for line in file:
            for j, value in enumerate(line.split(",")):
                kxky[j].append(np.float(value))
    kxky = np.asarray(kxky)
    kxky = kxky.T
    return kxky[:ledNum, :]


class AverageMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def np2tensor(x):
    return torch.from_numpy(x).permute(0, 3, 1, 2).float()


def initial_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform(m.weight)
            init.constant(m.bias, 0.0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1.0)
            init.constant(m.bias, 0.0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0.0)


def load_model(model, optimizer, saved_path):
    checkpoint = torch.load(saved_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def Nor(x):
    x -= x.min()
    x /= x.max()
    return x


def Nor2Pi(x):
    delta = x.max() - x.min()
    alpha = 2 * np.pi / delta
    tmp = x.min() * alpha + np.pi
    x = x * alpha - tmp
    return x


def model_test(args):
    if args.sample in ['U2OS', 'Hela']:
        from DFNN_biological import FP_SR_WDSR
        model = FP_SR_WDSR()
        model.cuda(0)
        opti = optim.Adam(model.parameters(), lr=1e-3, amsgrad=True, weight_decay=0.001)
        load_model(model, optimizer=opti, saved_path='model_weight/For_biological.tar')

        kxky = spiral_kxky('spiral_kxky.txt', 7 ** 2) * [1, -1]
        LR = np.ndarray((1, 49, 1800, 1800))

        if args.sample == 'U2OS':
            sample_path = r'data\test_7x7_U2OS'
            abs_v = [0.2, 0.9]
            pha_v = [-2.0, 2.0]
        elif args.sample == 'Hela':
            sample_path = r'data\test_7x7_vitroHela'
            abs_v = [0.2, 0.75]
            pha_v = [-1.6, 2.2]

        print('Reconstructing--->')
        for i in range(49):
            index = np.abs(kxky[i][1] - 3) * 7 + np.abs(kxky[i][0] + 4)
            LR[0, i,] = (io.imread(sample_path + r'\test ({}).tif'.format(int(index)), as_gray=True)[180:-180, 380:-380]) ** 1.0

        LR = torch.from_numpy(Nor(LR)).float().cuda()

        with torch.no_grad():
            SR = model(LR)

        plt.figure(), plt.imshow(Nor(SR.cpu()[0, 0]), cmap='gray', vmin=abs_v[0], vmax=abs_v[1])
        plt.figure(), plt.imshow(Nor2Pi(SR.cpu()[0, 1]), cmap='gray', vmin=pha_v[0], vmax=pha_v[1])

    elif args.sample == 'USAF':
        from DFNN_USAF import FP_SR_WDSR
        model = FP_SR_WDSR()
        model.cuda(0)
        opti = optim.Adam(model.parameters(), lr=1e-3, amsgrad=True, weight_decay=0.001)
        load_model(model, optimizer = opti, saved_path = 'model_weight/For_USAF.tar')

        kxky = spiral_kxky('spiral_kxky.txt', 11 ** 2) * [1, -1]
        LR = np.ndarray((1, 121, 128, 128))
        abs_v = [0.2, 0.9]
        pha_v = [0.0, 0.7]

        print('Reconstructing--->')
        for i in range(121):
            index = np.abs(kxky[i][1] - 5) * 11 + np.abs(kxky[i][0] + 6)
            LR[0, i,] = io.imread(r'data\test_11x11_USAF/test ({}).tif'.format(int(index)),
                                  as_gray=True)
        LR = torch.from_numpy(Nor(LR)).float().cuda()

        with torch.no_grad():
            SR = model(LR)

        plt.figure(), plt.imshow(Nor(SR.cpu()[0, 0]), cmap='gray', vmin=abs_v[0], vmax=abs_v[1])
        plt.figure(), plt.imshow(Nor(-SR.cpu()[0, 1]), cmap='gray', vmin=pha_v[0], vmax=pha_v[1])
    else:
        raise Exception('Invalid args.sample')


if __name__ == '__main__':

    cudnn.benchmark = True
    np.random.seed(1234)
    random.seed(1234)
    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)

    model_test(args)

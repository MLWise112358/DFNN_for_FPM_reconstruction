import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from System_parameter import args

W_N = lambda x: weight_norm(x)

class ResBlock_A(nn.Module):
    def __init__(self, n_feats, expansion_ratio, res_scale = 1.0):
        super(ResBlock_A, self).__init__()
        self.res_scale = res_scale
        self.module = nn.Sequential(
            W_N(nn.Conv2d(n_feats, n_feats*expansion_ratio, kernel_size = 3, padding = 1, stride = 1, dilation = 1, groups = 1, bias = True)),
            nn.ReLU(inplace = True),
            W_N(nn.Conv2d(n_feats*expansion_ratio, n_feats, kernel_size = 3, padding = 1, stride = 1, groups = 1, bias = True))
        )

    def forward(self, x):
        return x * self.res_scale + self.module(x)

class ResBlock_B(nn.Module):
    def __init__(self, n_feats, expansion_ratio, res_scale = 1.0, low_rank_ratio = 0.8):
        super(ResBlock_B, self).__init__()
        self.res_scale= res_scale
        self.module = nn.Sequential(
            W_N(nn.Conv2d(n_feats, n_feats*expansion_ratio, kernel_size = 1)),
            nn.ReLU(inplace = True),
            W_N(nn.Conv2d(n_feats*expansion_ratio, int(n_feats*low_rank_ratio), kernel_size = 1)),
            W_N(nn.Conv2d(int(n_feats*low_rank_ratio), n_feats, kernel_size = 3, padding = 1))
        )

    def forward(self, x):
        return x * self.res_scale + self.module(x)

class WDSR(nn.Module):
    def __init__(self, num, down_sample_scale, args = args):
        super(WDSR, self).__init__()
        self.Res_Block = self.block()
        if np.log2(down_sample_scale) == 0.0:
            down = [W_N(nn.Conv2d(args.num_LR, args.features, kernel_size=3, stride=1, padding=1))] #args.num_LR, args.features  #args.num_LR
        else:
            down = [W_N(nn.Conv2d(args.num_LR, args.features, kernel_size = 3, stride = 2, padding = 1))] + \
                   [W_N(nn.Conv2d(args.features, args.features, kernel_size = 3, stride = 2, padding = 1)) for _ in range(int(np.log2(down_sample_scale))-1)]
        body = [self.Res_Block for _ in range(num)]#args.num_res_block
        up = []
        for _ in range(int(np.log2(down_sample_scale))):
            up.append(W_N(nn.Conv2d(args.features, args.features * 4, kernel_size = 3, stride = 1, padding = 1)))
            up.append(nn.PixelShuffle(2))
        tail = [W_N(nn.Conv2d(args.features, 32 * args.scale ** 2, kernel_size = 3, stride = 1, padding = 1)),
                nn.PixelShuffle(args.scale)]

        self.down = nn.Sequential(*down)
        self.body = nn.Sequential(*body)
        self.up = nn.Sequential(*up)
        self.tail = nn.Sequential(*tail)

    def block(self):
        if args.model == 'WDSR_A':
            return ResBlock_A(args.features, expansion_ratio = 4, res_scale = args.res_scale) # 6
        elif args.model == 'WDSR_B':
            return ResBlock_B(args.features, expansion_ratio = 4, res_scale = args.res_scale, low_rank_ratio = args.low_rank_ratio) # 6
        else:
            raise Exception('invalid model style, need to be chose between \'WDSR_A\' and \'WDSR_B\'')

    def forward(self, x):
        x = self.down(x)
        x = self.body(x)
        x = self.up(x)
        x = self.tail(x)
        return x

class FP_SR_WDSR(nn.Module):
    def __init__(self):
        super(FP_SR_WDSR, self).__init__()
        up_1 = [
                W_N(nn.Conv2d(args.num_LR, args.features, kernel_size = 3, stride = 1, padding = 1)),
                nn.UpsamplingBilinear2d(scale_factor = 2),
                ]
        self.up_1 = nn.Sequential(*up_1)

        # ABS
        self.ABS1 = WDSR(num = 16, down_sample_scale = 1) #1
        self.ABS2 = WDSR(num = 16, down_sample_scale = 4) #2
        #PHASE
        self.PHASE1 = WDSR(num = 16, down_sample_scale = 1)
        self.PHASE2 = WDSR(num = 16, down_sample_scale = 4)
        self.Cat_abs = nn.Conv2d(64, 32, kernel_size = 3, padding = 1)
        self.out_abs = nn.Conv2d(32, 1, kernel_size = 1, padding = 0)
        self.Cat_phase = nn.Conv2d(64, 32, kernel_size = 3, padding = 1)
        self.out_phase = nn.Conv2d(32, 1, kernel_size =  1, padding = 0)

    def forward(self, x):

        #ABS
        ABS1 = self.ABS1(x)
        ABS2 = self.ABS2(x)
        Cat_abs = torch.cat([ABS1, ABS2], dim = 1)
        Cat_abs = self.Cat_abs(Cat_abs)
        out_abs = self.out_abs(Cat_abs)
        #PHASE
        PHASE1 = self.PHASE1(x)
        PHASE2 = self.PHASE2(x)
        Cat_phase = torch.cat([PHASE1, PHASE2], dim = 1)
        Cat_phase = self.Cat_phase(Cat_phase)
        out_phase = self.out_phase(Cat_phase)

        outp = torch.cat([out_abs, out_phase], dim = 1)

        return outp












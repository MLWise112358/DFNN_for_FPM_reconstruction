import argparse

class sys_peramater():
    def __init__(self, sam = None):
        self.sample = sam
        self.set_parameters()
    def set_parameters(self):
        if self.sample == 'USAF':
            self.array_size = 11
            self.features = 128
        elif self.sample in ['U2OS', 'Hela']:
            self.array_size = 7
            self.features = 64
        else:
            raise Exception('invalid sample')


sample = input('please input the sample type (USAF or U2OS or Hela):') #'biological' #'USAF', 'biological'
parser = argparse.ArgumentParser()
parser.add_argument('--sample', type = str, default = sample)
parser.add_argument('--model', type = str, default = 'WDSR_A') #WDSR_A, WDSR_B
parser.add_argument('--scale', type = int, default = 2) # 2
parser.add_argument('--features', type = int, default = sys_peramater(sample).features)
parser.add_argument('--num_res_block', type = int, default = 16)
parser.add_argument('--res_scale', type = float, default = 1.0)
parser.add_argument('--num_LR', type = int, default = sys_peramater(sample).array_size ** 2)

args = parser.parse_args()



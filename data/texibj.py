'''
#  filename: texibj.py
#  dataset for TexiBJ dataset
#  Likun Qin, 2021
'''
import h5py
import os
import numpy as np
import torch
import torch.utils.data as data

class TexiBJ(data.Dataset):
    '''
    dataset for TexiBJ

    '''
    def __init__(self, root, is_train, n_input=10, n_output=10):
        '''

        :param root: str or Path; directory of dataset
        :param is_train: boolean; it is training or testing?
        :param n_input: int; number of input
        :param n_output: int; number of gt

        -- root
            -- BJ13_M32x32_T30_InOut.h5
            -- BJ14_M32x32_T30_InOut.h5
            -- BJ15_M32x32_T30_InOut.h5
            -- BJ16_M32x32_T30_InOut.h5
        '''
        super(TexiBJ, self).__init__()
        self.root = root
        self.is_train = is_train
        self.n_input = n_input
        self.n_output = n_output
        self.files = ["BJ13_M32x32_T30_InOut.h5", "BJ14_M32x32_T30_InOut.h5",
                      "BJ15_M32x32_T30_InOut.h5", "BJ16_M32x32_T30_InOut.h5"]
        # the length of data in each file
        self.len = [4888, 4780, 5596, 7720]
        # the accumulation of the number of sequences
        self.acc = [4868, 9628, 15204, 22904]

        self.data = []
        self.date = []

        if self.is_train:
            self.length = 20904
            for one in self.files:
                f = h5py.File(os.path.join(self.root, one))
                self.data.append(f['data'])
                self.date.append(f['date'])
        else:
            self.length = 2000
            f = h5py.File(os.path.join(self.root, self.files[-1]))
            self.data.append(f['data'][-2020:, :, :, :])
            self.date.append(f['date'][-2020:])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        if self.is_train:
            if idx <= self.acc[0]:
                year_data = self.data[0]
                id = idx
            elif idx <= self.acc[1]:
                year_data = self.data[1]
                id = idx - self.acc[0]
            elif idx <= self.acc[2]:
                year_data = self.data[2]
                id = idx - self.acc[1]
            else:
                year_data = self.data[3]
                id = idx - self.acc[2]

            frames = year_data[id:id+20]  # [20, 2, 32, 32]

        else:
            frames = self.data[0][idx:idx+20]  # [20, 2, 32, 32]

        inputs = torch.from_numpy(frames[:10] / 1250).contiguous().float()  # [10, 2, 32, 32]
        outputs = torch.from_numpy(frames[10:] / 1250).contiguous().float()  # [10, 2, 32, 32]

        return [idx, inputs, outputs]


if __name__ == '__main__':
    dataset = TexiBJ(root='/home/cap/dataset/texibj', is_train=True)
    sample = dataset[4869]
    print('data get')
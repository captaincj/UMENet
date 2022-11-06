'''
#  filename: sst.py
#  dataset for SST
#  Likun Qin, 2021
'''
import os

import torch
import torch.utils.data as data
import numpy as np
import pickle


class SST(data.Dataset):
    '''
    dataset for SST
    '''
    def __init__(self, root, is_train, n_input=10, n_output=10):
        '''

        :param root: str or Path; the directory of SST dataset
        :param is_train: Boolean; whether it is for training or testing
        :param n_input: int; the number of known frames in a sequence
        :param n_output: int; the number of predictions in a sequence
        directory structure:
        -- root
            -- 2006
            -- 2019
            -- 2012
            -- 2015
        '''
        super(SST, self).__init__()
        self.root = root
        self.is_train = is_train

        assert n_input + n_output == 20
        self.n_input = n_input
        self.n_output = n_output

        self.file_list = []
        self.length = 0

        if is_train:
            self.folders = ['2006', '2009', '2012']
            for folder in self.folders:
                file_list = os.listdir(os.path.join(self.root, folder))
                file_list.sort()

                self.length += len(file_list)

                self.file_list.extend([os.path.join(folder, x) for x in file_list])


        else:
            self.folders = ['2015']
            file_list = os.listdir(os.path.join(self.root, self.folders[0]))
            file_list.sort()

            self.length += len(file_list)

            self.file_list.extend([os.path.join(self.folders[0], x) for x in file_list])

    def __len__(self):
        return self.length

    def __getitem__(self, item):

        with open(os.path.join(self.root, self.file_list[item]), 'rb') as f:
            data = pickle.load(f)

        seq = data['sequence']
        input = seq[:self.n_input]
        output = seq[self.n_input:]

        input = torch.from_numpy(input).contiguous().float()
        input = torch.unsqueeze(input, dim=1)
        output = torch.from_numpy(output).contiguous().float()
        output = torch.unsqueeze(output, dim=1)

        return [item, input, output]


if __name__ == '__main__':
    dataset = SST(root='/media/cap/Newsmy/dataset/sst/sst', is_train=True)
    print(len(dataset))
    sample = dataset.__getitem__(2000)
    print('get')






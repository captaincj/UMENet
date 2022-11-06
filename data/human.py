'''
#  filename: human.py
#  dataset for Human3.6m
#  require preprocessing data
#  Likun Qin, 2021
'''
import os
import torch
import torch.utils.data as data
import numpy as np
import cv2 as cv


class Human(data.Dataset):
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
            -- s1
                -- Direction
                ...
            -- s5
            -- s6
            -- s7
            -- s8
            -- s9
            -- s11
        '''
        super(Human, self).__init__()
        self.root = root
        self.is_train = is_train

        assert n_input + n_output == 20
        self.n_input = n_input
        self.n_output = n_output

        self.video_list = []  # store the name of each video
        self.file_list = []   # store the file list of each video
        self.length = 0
        acc = 0

        self.milestones = []   # store the accumulated sums

        if is_train:
            self.folders = ['s1', 's5', 's6', 's7', 's8' ]
        else:
            self.folders = ['s9', 's11']

        for folder in self.folders:
            video_list = os.listdir(os.path.join(self.root, folder))
            video_list.sort()

            # acc += len(video_list)
            # self.milestones.append(acc)

            # self.video_list.extend([os.path.join(folder, x) for x in video_list])
            for vid in video_list:
                self.video_list.append(os.path.join(folder, vid))
                file_list = os.listdir(os.path.join(root, folder, vid))
                file_list.sort()
                self.file_list.append(file_list)
                acc += (len(file_list) // 20 - 1)
                self.milestones.append(acc)

        self.length = acc

    def __len__(self):
        return self.length

    def __getitem__(self, item):

        start_id = 0
        video_id = 0
        for i, milestone in enumerate(self.milestones):
            if item <= milestone:
                if i != 0:
                    start_id = (item - self.milestones[i-1]) * 20
                    video_id = i

                else:
                    start_id = item * 20
                    video_id = 0

                break

        video = self.video_list[video_id]

        inputs = []
        outputs = []

        for j in range(start_id, start_id+20):
            img_name = self.file_list[video_id][j]
            img = cv.imread(os.path.join(self.root, video, img_name))
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # h, w, c
            img = np.transpose(img, (2, 0, 1))   # c, h, w

            if j - start_id < 10:
                inputs.append(torch.from_numpy(img / 255.0).contiguous().float())
            else:
                outputs.append(torch.from_numpy(img / 255.0).contiguous().float())


        inputs = torch.stack(inputs, dim=0)
        outputs = torch.stack(outputs, dim=0)

        return [item, inputs, outputs]


if __name__ == '__main__':
    dataset = Human(root='/media/cap/Newsmy/dataset/human', is_train=True)
    print(len(dataset))
    # get = [1,2]
    for i in range(100):
        sample = dataset.__getitem__(i)
        print(i, ' get')
'''
#  filename: sst_pre.py
#  preprocess SST dataset to convert nc to pkl
#  each .pk file store a dict:
#  save_dict = {
#                 'sequence': normalized sequence,
#                 'mean': mean,
#                 'std': std
#             }
#  Likun Qin, 2021
'''
# from netCDF4 import Dataset
from scipy.io import netcdf_file
import pickle
import os
import numpy as np
import copy

root = '/media/cap/Newsmy/dataset/sst'
save_root = '/media/cap/Newsmy/dataset/sst/pk'

files = [
         'global-reanalysis-phy-001-030-daily_06-09.nc',
         '2009-12-21-2012-12-10.nc',
         'global-reanalysis-phy-001-030-daily_2012-2015.nc',
         'global-reanalysis-phy-001-030-daily_2015-2017.nc'
        ]

year = ['2006', '2009', '2012', '2015']
# year = ['2015']

left_right_pts = [# the bottom layer
                  (60, 0), (188, 0), (252, 0), (316, 0), (380, 0), (444, 0), (510, 0),
                  (574,0), (638, 0),
                  # 2nd layer
                  (169, 64), (233, 64), (297, 64), (361, 64), (425, 64), (489, 64),
                  # 3rd layer
                  (278, 128), (344, 128), (408, 128), (472, 128), (536, 128), (600, 128),
                  # 4th layer
                  (387, 192), (451, 192), (515, 192), (579, 192),
                  # 5th layer
                  (387, 256), (451, 256), (515, 256), (579, 256), (633, 256),
                  # 6th layer
                  (515, 320), (579, 320), (633, 320),
                  ]  # total 28, (longitude, latitude)

for k in range(4):

    one = files[k]
    nc = netcdf_file(os.path.join(root, one), mmap=False, maskandscale=True)
    data = nc.variables['thetao']
    n_time = data.shape[0]
    yr = year[k]
    # num = 0

    for j, (x, y) in enumerate(left_right_pts):

        for i in range(0, n_time - 20, 20):
            seq = copy.deepcopy(data[i:i+20, 0, y:y+64, x:x+64])
            if np.any(seq.mask):
                print('location:', y, x, ' time ', i)
                print('masked!')
                # num += 1
            # mean = np.mean(seq)
            # std = np.std(seq)
            # res = (seq - mean) / std
            res = (seq + 3) / 36
            if np.any(res < 0):
                print('location:', y, x, ' time ', i)
                print('smaller than 0')
            if np.any(res > 1):
                print('location:', y, x, ' time ', i)
                print('bigger than 0')
            save_dict = {
                'sequence': res,
                # 'mean': mean,
                # 'std': std
            }

            with open(os.path.join(save_root, yr, yr + 'patch' + str(j) + '_n' + str(i) + '.pk'), 'wb') as f:
                pickle.dump(save_dict, f)

    nc.close()





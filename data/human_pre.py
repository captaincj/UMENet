'''
#  filename: human_pre.py
#  preprocess human3.6 dataset
#  1. resample
#  2. crop
'''
import cv2 as cv
import os

root = '/media/cap/Newsmy/dataset/human3.6/s11'
save = '/media/cap/Newsmy/dataset/human/s11'

if not os.path.exists(save):
    os.mkdir(save)

videos = os.listdir(root)
videos.sort()

for one in videos:
    if os.path.exists(os.path.join(save, one)):
        continue
    os.mkdir(os.path.join(save, one))
    img_list = os.listdir(os.path.join(root, one))
    img_list.sort()

    print(one)
    if '60457274' in one:
        vertical = 160
        horizontal = 160
    else:
        vertical = 200
        horizontal = 200

    print(vertical, ',', horizontal)

    # img0 = cv.imread(os.path.join(root, one, img_list[0]))
    # loop = 'y'
    # while loop == 'y':
    #     vertical = input('vertical:')
    #     vertical = int(vertical)
    #     horizontal = input('horizontal:')
    #     horizontal = int(horizontal)
    #     img_s = img0[vertical:-vertical, horizontal:-horizontal, :]
    #     cv.imshow('img', img_s)
    #     cv.waitKey()
    #     cv.destroyWindow('img')
    #     img_s = cv.resize(img_s, (128, 128))
    #     cv.imshow('small img', img_s)
    #     cv.waitKey()
    #     cv.destroyWindow('small img')
    #     loop = input('loop: (y or n)')

    for file in img_list:
        img = cv.imread(os.path.join(root, one, file))
        # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = img[vertical:-vertical, horizontal:-horizontal, :]
        img = cv.resize(img, (64,64))
        cv.imwrite(os.path.join(save, one, file), img)

# img = cv.imread('./000001.jpg')
# img = img[100:900, 100:900]
# # image = cv.resize(img, (64, 64))
# cv.imshow('img', img)
# cv.waitKey(3000)

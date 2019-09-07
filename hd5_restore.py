# -*- coding: utf-8 -*-
import h5py
import cv2
import numpy as np

# hd5_train_path = './data/train_morror_and_rotation.hd5'
hd5_train_path = './data/train_300w_org.hd5'
font = cv2.FONT_HERSHEY_SIMPLEX # font
def get_train_and_label(train_path):
    with h5py.File(train_path, 'r') as f:
        images = f['Images'][:]
        labels = f['landmarks'][:]
        oulaTheta = f['oulaTheta'][:]
    X_train = images
    Y_train = labels
    Y_theta = oulaTheta
    # print(X_train.shape) # (3111,112,3)
    #     # print(Y_train.shape) # (3111,136)
    #     # print(Y_theta.shape)
    return X_train,Y_train,Y_theta

X_train,Y1_train,Y2_train_Theta = get_train_and_label(hd5_train_path)
print(X_train.shape)
print(Y1_train.shape)
print(Y2_train_Theta.shape)

for i in range(len(X_train)):
    image = X_train[i] # 图像
    print(image.shape)
    print(Y1_train[i])
    marks = Y1_train[i]
    oula = Y2_train_Theta[i]
    print(oula)
    # cv2.imshow('image_%d' %i,image)
    marks = (marks + 0.5) * 112
    marks = np.reshape(marks,(-1,2))
    # sx, sy = 112 / crop.shape[0], 112 / crop.shape[1]
    for j in range(68):
        cv2.circle(image,(marks[j][0],marks[j][1]),1,(255,0,255))
    cv2.putText(image,'pitch:%0.3f' % oula[0],(2,2),font,0.5,(255,255,0))
    cv2.putText(image, 'roll:%0.3f' % oula[1], (6, 2), font, 0.5, (255, 255, 0))
    cv2.putText(image, 'yaw:%0.3f' % oula[2], (10, 2), font, 0.5, (255, 255, 0))
    cv2.imwrite('./data/restore/restore_%d.jpg' %i,image)
    cv2.waitKey(200)





# -*- coding: utf-8 -*-
# 之前测试达到9.18%的最优模型--数据集：train_300w_org.hd5
# blocknet:输入加bn
import tensorflow as tf
from PFLD_pre import weight_variable,bias_variable,conv2d,deepwise_conv2d,make_bottleneck_block
from PFLD_pre import get_train_and_label,crop_and_concat,batch_norm
import os
import random
import numpy as np
from sklearn.model_selection import train_test_split # 数据集拆分

#  主网络 + 副网络
def Pfld_Netework(input): # 112 * 112 * 3
    with tf.name_scope('Pfld_Netework'):
        ##### Part1: Major Network -- 主网络 #####
        #layers1
        #input= [None,112,112,3]
        with tf.name_scope('layers1'):
            W_conv1 = weight_variable([3, 3, 3, 64],name='W_conv1')
            b_conv1 = bias_variable([64],name='b_conv1')
            x_image = tf.reshape(input, [-1, 112, 112, 3],name='input_X')
            x_image = batch_norm(x_image,is_training=True)
            h_conv_1 = conv2d(x_image,W_conv1,strides=[1,2,2,1],padding='SAME') + b_conv1
        # layers2
        with tf.name_scope('layers1'):
            W_conv2 = weight_variable([3,3,64,1],name='W_conv2')
            b_conv2 = bias_variable([64],name='b_conv2')
            h_conv_1 = batch_norm(h_conv_1,is_training=True)
            h_conv_2 = deepwise_conv2d(h_conv_1,W_conv2) + b_conv2 # 56 * 56 * 64
        # Bottleneck   input = [56*56*64]
        with tf.name_scope('Mobilenet-V2'):
            with tf.name_scope('bottleneck_1'):
                h_conv_b1 = make_bottleneck_block(h_conv_2, 2, 64, stride=[1, 2, 2, 1], kernel=(3, 3)) # 28*28*64
                h_conv_b1 = make_bottleneck_block(h_conv_b1, 2, 64, stride=[1, 1, 1, 1], kernel=(3, 3))  # 28*28*64
                h_conv_b1 = make_bottleneck_block(h_conv_b1, 2, 64, stride=[1, 1, 1, 1], kernel=(3, 3))  # 28*28*64
                h_conv_b1 = make_bottleneck_block(h_conv_b1, 2, 64, stride=[1, 1, 1, 1], kernel=(3, 3))  # 28*28*64
                h_conv_b1 = make_bottleneck_block(h_conv_b1, 2, 64, stride=[1, 1, 1, 1], kernel=(3, 3))  # 28*28*64
            with tf.name_scope('bottleneck_2'):
                h_conv_b2 = make_bottleneck_block(h_conv_b1,2,128,stride=[1,2,2,1],kernel=(3,3)) # 14*14*128
            with tf.name_scope('bottleneck_3'):
                h_conv_b3 = make_bottleneck_block(h_conv_b2, 4, 128, stride=[1, 1, 1, 1], kernel=(3, 3)) # 14*14*128
                h_conv_b3 = make_bottleneck_block(h_conv_b3, 4, 128, stride=[1, 1, 1, 1], kernel=(3, 3))  # 14*14*128
                h_conv_b3 = make_bottleneck_block(h_conv_b3, 4, 128, stride=[1, 1, 1, 1], kernel=(3, 3))  # 14*14*128
                h_conv_b3 = make_bottleneck_block(h_conv_b3, 4, 128, stride=[1, 1, 1, 1], kernel=(3, 3))  # 14*14*128
                h_conv_b3 = make_bottleneck_block(h_conv_b3, 4, 128, stride=[1, 1, 1, 1], kernel=(3, 3))  # 14*14*128
                h_conv_b3 = make_bottleneck_block(h_conv_b3, 4, 128, stride=[1, 1, 1, 1], kernel=(3, 3))  # 14*14*128
            with tf.name_scope('bottleneck_4'):
                h_conv_b4 = make_bottleneck_block(h_conv_b3,2,16,stride=[1,1,1,1],kernel=(3,3))  # 14*14*16
        # S1
        with tf.name_scope('S1'):
            h_conv_s1 = h_conv_b4 # 14 * 14 * 16
        # s2
        with tf.name_scope('S2'):
            W_conv_s2 = weight_variable([3,3,16,32],name='W_conv_s2')
            b_conv_s2 = bias_variable([32],name='b_conv_s2')
            h_conv_s1 = batch_norm(h_conv_s1, is_training=True)
            h_conv_s2 = conv2d(h_conv_s1,W_conv_s2,strides=[1,2,2,1],padding='SAME') + b_conv_s2  # 7*7*32
        # S3
        with tf.name_scope('S3'):
            W_conv_s3 = weight_variable([7,7,32,128],name='W_conv_s3')
            b_conv_s3 = bias_variable([128],name='b_conv_s3')
            h_conv_s2 = batch_norm(h_conv_s2, is_training=True)
            h_conv_s3 = conv2d(h_conv_s2,W_conv_s3,strides=[1,1,1,1],padding='VALID') + b_conv_s3  # 1 * 1 * 128

        # MS-FC(多尺度全连接层)
        with tf.name_scope('MS-FC'):
            ########################111--收敛较快#################################
            W_conv_fc_s1 = weight_variable([14, 14, 16, 64], name='W_conv_s2')
            b_conv_fc_s1 = bias_variable([64], name='b_conv_s2')
            h_conv_s1 = batch_norm(h_conv_s1,is_training=True)
            h_conv_fc_s1 = conv2d(h_conv_s1,W_conv_fc_s1,strides=[1,1,1,1],padding='VALID') + b_conv_fc_s1 # 1*1*64

            W_conv_fc_s2 = weight_variable([7, 7, 32, 64], name='W_conv_s2')
            b_conv_fc_s2 = bias_variable([64], name='b_conv_s2')
            h_conv_s2 = batch_norm(h_conv_s2, is_training=True)
            h_conv_fc_s2 = conv2d(h_conv_s2, W_conv_fc_s2, strides=[1, 1, 1, 1],padding='VALID') + b_conv_fc_s2  # 1*1*64

            h_conv_s3 = batch_norm(h_conv_s3, is_training=True)
            h_conv_ms_fc = tf.concat([h_conv_fc_s1,h_conv_fc_s2,h_conv_s3],axis=3) # 1*1*256
            h_conv_ms_fc = tf.reshape(h_conv_ms_fc,(-1,1*1*256))
            W_ms_fc = weight_variable([1*1*256, 136], name='W_fc_s2')
            b_ms_fc = bias_variable([136], name='b_fc_s2')
            pre_landmark = tf.add(tf.matmul(h_conv_ms_fc, W_ms_fc), b_ms_fc, name='landmark_3')

            #########################222--效果很差################################
            """
            W_conv_fc_s1 = tf.reshape(h_conv_s1,[-1,14*14*16])
            W_conv_fc_s2 = tf.reshape(h_conv_s2,[-1,7*7*32])
            W_conv_fc_s3 = tf.reshape(h_conv_s3,[-1,1*1*128])
            h_conv_ms_fc = tf.concat([W_conv_fc_s1,W_conv_fc_s2,W_conv_fc_s3],axis=1)
            h_conv_ms_fc1 = tf.reshape(h_conv_ms_fc,(-1,1*1*4832))
            # W_ms_fc0 = weight_variable([1*1*4832, 1024], name='W_fc_s2')
            # b_ms_fc0 = bias_variable([1024], name='b_fc_s2')
            # pre_land_mark0 = tf.add(tf.matmul(h_conv_ms_fc1,W_ms_fc0),b_ms_fc0)
            W_ms_fc = weight_variable([1 * 1 * 4832, 136], name='W_fc_s2')
            b_ms_fc = bias_variable([136], name='b_fc_s2')
            h_conv_ms_fc1 = batch_norm(h_conv_ms_fc1,is_training=True)
            pre_landmark = tf.add(tf.matmul(h_conv_ms_fc1, W_ms_fc), b_ms_fc, name='landmark_3')
            """
            ######################### 333最初用的 ################################
            """
            concat1 = crop_and_concat(h_conv_s1,h_conv_s2) # (?,7,7,78)
            concat2 = crop_and_concat(concat1,h_conv_s3) #(?,1,1,176)
            h_conv_ms_fc = tf.reshape(concat2, [-1, 1 * 1 * 176])
            W_ms_fc = weight_variable([1 * 1 * 176, 136], name='W_fc_s2')
            b_ms_fc = bias_variable([136], name='b_fc_s2')
            pre_landmark = tf.add(tf.matmul(h_conv_ms_fc, W_ms_fc), b_ms_fc, name='landmark_3')
            """
            ##### Part2: Auxiliary Network -- 副网络 #####
        # layers1
        # 副网络输入input： h_conv_b1 === [1,28,28,64]
        with tf.name_scope('Funet-layers1'):
            W_convfu_1 = weight_variable([3, 3, 64, 128],name='W_convfu_1')
            b_convfu_1 = bias_variable([128],name='b_convfu_1')
            h_convfu_1 = conv2d(h_conv_b1, W_convfu_1,strides=[1,2,2,1],padding='SAME') + b_convfu_1  # 14 * 14 * 128
        # layers2
        with tf.name_scope('Funet-layers2'):
            W_convfu_2 = weight_variable([3,3,128,128],name='W_convfu_2')
            b_convfu_2 = bias_variable([128],name='b_convfu_2')
            h_convfu_2 = conv2d(h_convfu_1,W_convfu_2,strides=[1,1,1,1],padding='SAME') + b_convfu_2 # 14 * 14 * 128
        # layers3
        with tf.name_scope('Funet-layers3'):
            W_convfu_3 = weight_variable([3,3,128,32],name='W_convfu_3')
            b_convfu_3 = bias_variable([32],name='b_convfu_3')
            h_convfu_3 = conv2d(h_convfu_2,W_convfu_3,strides=[1,2,2,1],padding='SAME') + b_convfu_3 # 7 * 7 * 32
        # layers4
        with tf.name_scope('Funet-layers4'):
            W_convfu_4 = weight_variable([7,7,32,128],name='W_convfu_4')
            b_convfu_4 = bias_variable([128],name='b_convfu_4')
            h_convfu_4 = conv2d(h_convfu_3,W_convfu_4,strides=[1,1,1,1],padding='VALID') + b_convfu_4 # 1 *  1 * 128
        ####### Fc ######
        # Fc1:
        with tf.name_scope('Fc1'):
            W_fu_fc1 = weight_variable([1 * 1 * 128, 32],name='W_fu_fc1')
            b_fc_s1 = bias_variable([32],name='b_fc_s1')
            h_convfu_4_fc = tf.reshape(h_convfu_4, [-1, 1 * 1 * 128])   # 1 * 128
            pre_theat_s1 = tf.matmul(h_convfu_4_fc, W_fu_fc1) + b_fc_s1 # 1 * 32
        # Fc2:
        with tf.name_scope('Fc2'):
            W_fu_fc2 = weight_variable([1 * 1 *32, 3],name='W_fu_fc2')
            b_fc_s2 = bias_variable([3],name='b_fc_s2')
            h_convfu_5_fc = tf.reshape(pre_theat_s1, [-1, 1 * 1 * 32]) # 1 * 32
            pre_theat = tf.add(tf.matmul(h_convfu_5_fc, W_fu_fc2),b_fc_s2,name='pre_theta')   # 1 * 3

    return pre_landmark,pre_theat

image_size = 112
# hd5_train_path = './data/train_300w_org.hd5' # 扩充了的数据集
# X_train,Y1_train,Y2_train_Theta = get_train_and_label(hd5_train_path)
# image = X_train[0:1,:,:,:]
# label1 = Y1_train[0:1,:]
# print(image.shape) # (1,112,112,3)
# print(label1.shape)
# print(label1)
# # 定义占位符
with tf.name_scope('input'):
    #X = image
    X = tf.placeholder(tf.float32,[None,image_size,image_size,3],name='X') # 输入训练图像

with tf.name_scope('out_put'):
    Y1 = tf.placeholder(tf.float32,[None,136],name='Y1') # 输入训练label1--land_mark
    Y2 = tf.placeholder(tf.float32,[None,3],name='Y2')   # 输入训练label2--theta

# 网络输出结果
pre_land_mark,pre_theat = Pfld_Netework(X)

# step:
global_step = tf.Variable(0,trainable=False)
# learning:指数衰减 epoch:100 衰减0.9 ---》指数衰减
learning_rate = tf.train.exponential_decay(0.0001,global_step=global_step,decay_steps=100000,decay_rate=1.0)
# l2正则化
l2_reg = tf.contrib.layers.apply_regularization(
                      tf.contrib.layers.l2_regularizer(1e-10), tf.trainable_variables())
# loss
loss1 = tf.sqrt(tf.reduce_mean(tf.square(Y1 - pre_land_mark)))
#loss1 = tf.losses.mean_squared_error(Y1,pre_land_mark) + l2_reg
loss2 = tf.losses.mean_squared_error(Y2,pre_theat)
total_loss = loss1 + loss2

# optimizer
optimizer = tf.contrib.layers.optimize_loss(total_loss, global_step=global_step,
                                            learning_rate=learning_rate,
                                            optimizer='Adam', increment_global_step=True)

############################ 获取训练集/验证集 #############################
# hd5_train_path = './data/train_mirror_and_rotation.hd5' # train_path
hd5_train_path = './data/train_300w_hudu.hd5' # 扩充了的数据集
test_size = 0.1 # 验证集比例
# 获取数据
X_train,Y1_train,Y2_train_Theta = get_train_and_label(hd5_train_path)
# 拆分训练集和验证集
X_train, X_valid, Y1_train, Y1_valid,Y2_train,Y2_valid \
    = train_test_split(X_train, Y1_train,Y2_train_Theta, test_size=test_size)
y_dim = Y1_train.shape[1] # 136 (68 * 2)
############################ Training Model ##############################
with tf.Session() as sess:
    print('---start train...---')
    init = tf.global_variables_initializer()
    sess.run(init)
    OUTPUT_DIR = './model'
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    tf.summary.scalar('loss/loss1',loss1)
    tf.summary.scalar('loss/loss2',loss2)
    tf.summary.scalar('loss/total_loss', total_loss)
    summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(OUTPUT_DIR, sess.graph)

    loss_valid_min = np.inf  # 验证集loss初始最小值
    saver = tf.train.Saver()
    epoch = 101  # 总的训练轮数
    batch_size = 32  # batch_size
    # 使用earning stoping
    patience = 10 # 耐心度
    for e in range(epoch):  # 每一轮迭代--都随机打乱数据
        loss_train = []  # 训练损失
        loss_valid = []  # 验证损失
        loss1_train = []
        loss2_train = []
        # 计算数据集上完成一轮迭代需要的批次数
        epoch_train_batch = (X_train.shape[0] // batch_size)
        # ############ 训练集/验证集随机打乱顺序 #######
        # 训练集
        idx_list_train = np.arange(X_train.shape[0])
        random.shuffle(idx_list_train) # 按照相同顺序打乱数据集
        X_train,Y1_train,Y2_train = X_train[idx_list_train,:],Y1_train[idx_list_train,:],Y2_train[idx_list_train,:]
        # 验证集
        idx_list_valid = np.arange(X_valid.shape[0])
        random.shuffle(idx_list_valid)
        X_valid,Y1_valid,Y2_valid = X_valid[idx_list_valid,:],Y1_valid[idx_list_valid,:],Y2_valid[idx_list_valid,:]
        ###########################################
        # 每一个batch_size(批数据)
        for i in range(epoch_train_batch): # 第i批数据
            X_batch =  X_train[i * batch_size:(i + 1) * batch_size, :, :, :]
            Y1_batch = Y1_train[i * batch_size:(i + 1) * batch_size, :]
            Y2_batch = Y2_train[i * batch_size:(i + 1) * batch_size, :]
            _,ls,ls1,ls2,other_loss,lr = sess.run([optimizer,total_loss,loss1,loss2,summary,learning_rate],
                       feed_dict={X:X_batch,Y1:Y1_batch,Y2:Y2_batch})
            loss_train.append(ls)
            loss1_train.append(ls1)
            loss2_train.append(ls2)
            if i > 0 and i % 20 == 0:
                writer.add_summary(sess.run(summary,feed_dict={X:X_batch,Y1:Y1_batch,Y2:Y2_batch}),
                                   e * X_train.shape[0] // batch_size + i)
                writer.flush()
        loss_train = np.mean(loss_train) # 每一个epoch后的loss均值
        loss1_train = np.mean(loss1_train)
        loss2_train = np.mean(loss2_train)
        # 验证集
        for j in range(X_valid.shape[0] // batch_size):
            X_valid_batch = X_valid[j * batch_size: (j + 1) * batch_size, :, :, :]
            Y1_valid_batch = Y1_valid[j * batch_size: (j + 1) * batch_size, :]
            Y2_valid_batch = Y2_valid[j * batch_size: (j + 1) * batch_size, :]
            ls_val = sess.run(total_loss,feed_dict={X:X_valid_batch,Y1:Y1_valid_batch,Y2:Y2_valid_batch})
            loss_valid.append(ls_val)
        loss_valid  = np.mean(loss_valid)
        print('Epoch %d,lr %.6f,train loss %.6f,valid loss %.6f' % (e, lr, loss_train, loss_valid))
        print('--- loss1:%s  ---*****---  loss2:%s' % (loss1_train,loss2_train))
        # #判断
        # if loss_valid < loss_valid_min:
        #     print('Saving model......')
        #     saver.save(sess,os.path.join(OUTPUT_DIR,'face_mark'))
        #     loss_valid_min = loss_valid
        #     patience = 10
        # else:
        #     patience -= 1
        #     if patience == 0:
        #         # 连续20次验证集loss不下降,退出训练
        #         break
    print('Saving model...---after train %s epochs'%epoch)
    saver.save(sess, os.path.join(OUTPUT_DIR, 'face_mark-100'))

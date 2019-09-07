import tensorflow as tf
import cv2
import os
import numpy as np

Mean_train_set = [127.5,127.5,127.5] # 图像均值
Scale_train_set = 0.0078125
Norm_image_size = 112 # 图像大小

font = cv2.FONT_HERSHEY_SIMPLEX # font
OUTPUT_DIR = 'model/'
model_read_path = os.path.join(OUTPUT_DIR,'face_mark_shuffle-200.meta') # model_path

with tf.Session() as sess:
   saver = tf.train.import_meta_graph(model_read_path) # load graph
   saver.restore(sess,tf.train.latest_checkpoint(OUTPUT_DIR))
   graph = tf.get_default_graph()
   X = graph.get_tensor_by_name('input/X:0') # 网络输入 (112,112,3)
   output_y = graph.get_tensor_by_name('Pfld_Netework/Fc-s3/landmark_3:0')  # 预测的关键点
   output_theta = graph.get_tensor_by_name('Pfld_Netework/Fc2/pre_theta:0') # 预测的欧拉角
   # 人脸检测器
   face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
   # 开启摄像头
   cap = cv2.VideoCapture(0)
   while True:
       ret, image = cap.read()
       if ret:
           #image = cv2.imread('./test_img/000.jpg')
           faces = face_cascade.detectMultiScale(image,1.1,10)
           if len(faces) > 0:# 如果检测到人脸
               for faceRect in faces:
                    x,y,w,h = faceRect # 人脸框 x,y,w,h = (146,270) (637,761)
                    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,255))
                    crop = image[y:y+h,x:x+w] #人脸框对应的图像[y0:y1,x0:x1]
                    # print(crop.shape)
                    sx,sy = 112/crop.shape[0],112/crop.shape[1]
                    # print(crop.shape)
                    # cv2.imshow('image', crop)
                    crop1 = cv2.resize(crop,(112,112),interpolation=cv2.INTER_CUBIC)
                    image_in = crop1.astype('f4')
                    image_chs = cv2.split(image_in)  # 拆分通道
                    for k, img_ch in enumerate(image_chs):
                        image_chs[k] = (img_ch - Mean_train_set[k]) * Scale_train_set
                    input = cv2.merge(image_chs)  # 合并通道---输入网络的图片 (112,112,3)

                    # 关键点检测
                    pre_mark = sess.run(output_y,feed_dict={X: np.reshape(input,[-1,112,112,3])})
                    # print(pre_mark)
                    pre_theta = sess.run(output_theta, feed_dict={X: np.reshape(input, [-1, 112, 112, 3])})
                    # 反归一化
                    marks = (pre_mark + 0.5) * 112

                    marks = np.reshape(marks,(-1,2))
                    for j in range(68):
                        # 反缩放 + 人脸剪裁原点
                        pre_x = int(marks[j][0] / sx + x)
                        # pre_x = int(marks[j][0] + x)
                        # pre_y = int(marks[j][1] + y)
                        pre_y = int(marks[j][1] / sy + y)
                        cv2.circle(image, (pre_x, pre_y), 1, (255,255, 0), -1)
                        cv2.circle(image, (pre_x, pre_y), 2, (0, 255,255), -1)
                    cv2.putText(image, 'pitch: %0.3f' % pre_theta[0][0], (400, 80), font, 0.5, (0, 0, 255))
                    cv2.putText(image, 'roll:  %0.3f' % pre_theta[0][1], (400, 95), font, 0.5, (0, 0, 255))
                    cv2.putText(image, 'yaw:   %0.3f' % pre_theta[0][2], (400, 110), font, 0.5, (0, 0, 255))
               cv2.imshow('face_landmark',image)
               if cv2.waitKey(1) & 0xFF == ord('q'):
                   break
   cap.release()
   cv2.destroyAllWindows()







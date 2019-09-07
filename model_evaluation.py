# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import tensorflow as tf

DEBUG = True

def load_model(Model_DIR,model_path,pred_pt_name,pred_theta_name):
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(model_path)  # load graph
        saver.restore(sess, tf.train.latest_checkpoint(Model_DIR + '/./'))
        graph = tf.get_default_graph()
        X = graph.get_tensor_by_name('input/X:0')  # 网络输入 (112,112,3)
        pred_pt = graph.get_tensor_by_name(pred_pt_name)  # 预测的关键点land_mark_3
        pred_theta = graph.get_tensor_by_name(pred_theta_name)  # 预测的欧拉角
        return X,pred_pt,pred_theta,sess

def read_landmarktxt(file_path):
    landmarks_dict = {}
    bbox_dict = {}
    with open(file_path,'r') as f:
        lines = [p.strip() for p in f.readlines()]
    for k,row in enumerate(lines):
        parts = row.split(' ')
        filename = parts[0]
        bbox = [float(parts[1]),float(parts[2]),float(parts[3]),float(parts[4])]
        pts = []
        for i in range(68):
            pts.append([float(parts[i*2+5]),float(parts[i*2+6])])
        assert len(pts) == 68
        landmarks_dict[filename] = pts
        bbox_dict[filename] = bbox
    return bbox_dict,landmarks_dict


def crop_face(img, det_box, gt_landmark):
    """
    剪裁人脸框位置
    :param img: image
    :param det_box: 人脸框位置
    :param gt_landmark: 关键点位置
    :return: crop_face，crop_landmark，crop_bbox,crop_offset
    """
    # 扩充det_box 上下左右各1０％ s = 0.1
    # 扩充det_box 上下左右各15％ s = 0.15
    # 得到标注关键点最大最小值
    all_x = []
    all_y = []
    for i in range(68):
        all_x.append(gt_landmark[i][0])
        all_y.append(gt_landmark[i][1])
    x_min = min(all_x)
    x_max = max(all_x)
    y_min = min(all_y)
    y_max = max(all_y)
    # 修改扩充框区域
    s = 0.1
    width = det_box[2]
    height = det_box[3]

    new_top = det_box[1]
    new_bottom = det_box[1] + det_box[3] +  s * height
    new_left = det_box[0] -  s * width
    new_right = det_box[0] + det_box[2] +  s * width
    new_top = new_top + 0.6 * (height - width)

    new_top = np.maximum(0, new_top)
    new_left = np.maximum(0, new_left)
    new_bottom = np.minimum(img.shape[0], new_bottom)
    new_right = np.minimum(img.shape[1], new_right)

    # 原人脸框扩充0.1后对应的人脸框
    # 加判断--如果有关键点在人脸扩充框之外 -- 设置/强行设置最小外接矩形
    if new_left > x_min or new_top > y_min or new_right < x_max or new_bottom < y_max:
        # 有关键点在人脸扩充框之外
        if new_top > y_min:
            new_top = y_min - 0.08 * height  # 防止top点正好在线上
            if new_top < 0:
                new_top = 0
        if new_left > x_min:
            new_left = x_min - 0.07 * width  # 根据最小关键点坐ｘ标再往外扩充0.05
            if new_left < 0:  # 边界判断
                new_left = 0
        if new_right < x_max:
            new_right = x_max + 0.07 * width
            if new_right > img.shape[1]:
                new_right = img.shape[1]
        if new_bottom < y_max:
            new_bottom = y_max + 0.05 * height
            if new_bottom > img.shape[0]:
                new_bottom = img.shape[0]
    ''' 裁剪之后的人脸 '''
    crop_face = img[int(new_top):int(new_bottom), int(new_left):int(new_right), :]
    ''' 裁剪之后人脸的关键点坐标位置，以裁剪图像左上角为原点'''
    crop_landmark = [[p[0] - new_left, p[1] - new_top] for p in gt_landmark]
    ''' 裁剪之后图像的SSD检测框位置，以裁剪图像左上角为原点'''
    crop_bbox = [det_box[0]-new_left, det_box[1]-new_top, det_box[2], det_box[3]]
    ''' 裁剪图像相对于原图的偏移量'''
    crop_offset = [new_left, new_top]
    return crop_face, crop_landmark, crop_bbox, crop_offset


def predict_landmarks(sess,crop_face,crop_offset,X,pred_pt,norm_size=112):
    #predict_landmarks(sess,crop_face,crop_offset,X,pred_pt,pred_theta,norm_size=112)
    resize_face = cv2.resize(crop_face,(norm_size,norm_size)) # (112,112,3)
    # 与训练时做相同预处理
    meanTrainSet = [127.5,127.5,127.5]
    scaleTrainSet = 0.00784313
    image = resize_face.astype('f4')
    image_chs = cv2.split(image)  # 拆分通道
    for k, img_ch in enumerate(image_chs):
        image_chs[k] = (img_ch - meanTrainSet[k]) * scaleTrainSet  # 图像归一化到 -- [-1,1]
        # image_chs[k] = (img_ch - meanTrainSet[k]) # 这里只做减均值
    input = cv2.merge(image_chs)  # 合并通道 (112,112,3)
    # with tf.Session() as sess:
    #     saver = tf.train.import_meta_graph(model_path)  # load graph
    #     saver.restore(sess, tf.train.latest_checkpoint(Model_DIR + '/./'))
    #     graph = tf.get_default_graph()
    #     X = graph.get_tensor_by_name('input/X:0')  # 网络输入 (112,112,3)
    #     pred_pt = graph.get_tensor_by_name(pred_pt_name)  # 预测的关键点land_mark_3
    #     pred_theta = graph.get_tensor_by_name(pred_theta_name)  # 预测的欧拉角

    # 关键点检测
    pre_mark = sess.run(pred_pt, feed_dict={X: np.reshape(input, [-1, 112, 112, 3])})
    #print(pre_mark.shape) # (batch_size,136)
    #pre_theta = sess.run(pred_theta, feed_dict={X: np.reshape(input, [-1, 112, 112, 3])})
    assert pre_mark.shape[1] == 136
    pred_landmarks = np.array([[pre_mark[0][i * 2],pre_mark[0][i * 2 + 1]] for i in range(68)])
    crop_offset_x = crop_offset[0]
    crop_offset_y = crop_offset[1]
    "将预测的结果转换回到图像坐标"
    map_to_src_landmarks = np.zeros(pred_landmarks.shape)
    #print(pred_landmarks.shape) # (68,2)
    sy, sx = crop_face.shape[0] / 112, crop_face.shape[1] / 112  # 从crop 到 112的缩小倍数
    for i in range(68):
        # map_to_src_landmarks[i][0] = (pred_landmarks[i][0] + 0.5) * crop_face.shape[1] + crop_offset_x
        # map_to_src_landmarks[i][1] = (pred_landmarks[i][1] + 0.5) * crop_face.shape[0] + crop_offset_y
        map_to_src_landmarks[i][0] = (pred_landmarks[i][0] + 0.5) * 112 * sx + crop_offset_x
        map_to_src_landmarks[i][1] = (pred_landmarks[i][1] + 0.5) * 112 * sy + crop_offset_y
    # show_result
    show_img = crop_face.copy()
    pred_point = map_to_src_landmarks
    for i in range(68):
        cv2.circle(show_img, (int(pred_point[i][0] - crop_offset_x), int(pred_point[i][1] - crop_offset_y)),
                   3, (0, 0, 255),-1)
    return map_to_src_landmarks,show_img



def calculate_NME(true_pts,pred_pts,target_parts=None,normalization='centers',showResults=False,verbose=False):
    assert true_pts.shape[0] == 68
    assert true_pts.shape[1] == 2
    assert true_pts.shape == pred_pts.shape
    if normalization == 'centers':
        # 眼睛瞳孔间的距离
        # print(np.mean(true_pts[[43,44,46,47],:],axis=0))
        # print(np.mean(true_pts[[37,38,40,41],:],axis=0))
        normDist = np.linalg.norm(np.mean(true_pts[[43,44,46,47],:],axis=0) - np.mean(true_pts[[37,38,40,41],:],axis=0))
    elif normalization == 'corners':
        # 左右眼外眼角的距离
        normDist = np.linalg.norm(true_pts[36] - true_pts[45])
    elif normalization == 'diagonal':
        # 标注关键点外接矩形框的对角线距离
        height,width = np.max(true_pts,axis=0) - np.min(true_pts,axis=0)
        normDist = np.sqrt(width ** 2 + height ** 2)
    if target_parts is None:
        error = np.mean(np.sqrt(np.sum((true_pts - pred_pts)**2,axis=1))) / normDist
        error = error * 100
        return error
    else:
        target_true_pts = true_pts[target_parts,:]
        target_pred_pts = pred_pts[target_parts,:]
        error = np.mean(np.sqrt(np.sum((target_true_pts - target_pred_pts)**2,axis=1))) / normDist
        error = error * 100
        return error


def model_eval():
    img_dir = './test/3-19-300W-common-test/helen&lfpw-testimage' # 测试图片集
    gt_file = './test/3-19-300W-common-test/helen&lfpw-bbox+landmark/test.txt'   # 标注文件路径
    #Model_DIR = 'model/4.23/test_1'  # model_path
    Model_DIR = 'model/'
    model_path = os.path.join(Model_DIR, 'face_mark-100.meta')  # model_name
    save_dir = './test_result/'
    # pred_pt = graph.get_tensor_by_name('Pfld_Netework/Fc-s1/landmark_1:0')  # 预测的关键点land_mark_1
    # pred_pt = graph.get_tensor_by_name('Pfld_Netework/Fc-s2/landmark_2:0')  # 预测的关键点land_mark_2
    # ored_pt = graph.get_tensor_by_name('Pfld_Netework/Fc-s3/landmark_3:0')  # 预测的关键点land_mark_2
    #output_pt_layer_name = 'Pfld_Netework/Fc-s3/landmark_3:0'
    output_pt_layer_name = 'Pfld_Netework/MS-FC/landmark_3:0'
    #output_theta_layer_name = 'Pfld_Netework/Fc2/pre_theta:0'
    image_size = 112
    # 区域
    contour_idx_list = range(0, 17)     # 脸轮廓-- 17
    eyebrow_idx_list = range(17, 27)    # 眉毛---- 10
    eye_idx_list = range(36, 48)        # 眼睛---- 12
    nose_idx_list = range(27, 36)       # 鼻子---- 9
    mouth_idx_list = range(48, 68)      # 嘴巴-----20
    nothing_face_idx_list = range(18,68)# 无脸轮廓--50

    # point,bbox
    bbox_dict,pts_dict = read_landmarktxt(gt_file)
    img_list = os.listdir(img_dir)
    img_list = [p for p in img_list if p.find('.jpg') > 0 or p.find('.jpeg') > 0]
    error_all_list = []
    error_eye_list = []
    error_nose_list = []
    error_mouth_list = []
    error_brow_list = []
    error_contour_list = []
    error_nothing_face_list = []
    outside_landmarks_cnt = 0
    #### load_model ####
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(model_path)  # load graph
        saver.restore(sess, tf.train.latest_checkpoint(Model_DIR + '/./'))
        graph = tf.get_default_graph()
        X = graph.get_tensor_by_name('input/X:0')  # 网络输入 (112,112,3)
        pred_pt = graph.get_tensor_by_name(output_pt_layer_name)  # 预测的关键点land_mark_3
        #pred_theta = graph.get_tensor_by_name(output_theta_layer_name)  # 预测的欧拉角
        #对图片库的每一张测试图片
        for name in img_list:
            img_path = os.path.join(img_dir,name)
            img = cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),-1)
            true_bbox = bbox_dict[name] # 获取该图片的bbox框
            true_pts = np.array(pts_dict[name]) # 获取该图片的point
            ''' check if main internal points are all within the given bbox.'''
            left, top0 = true_pts[36]  # 左眼外眼角点
            right, top1 = true_pts[45] # 右眼外眼角点
            mid, bot = true_pts[57]    # 下嘴唇中点
            top = max([top0, top1])
            left = min([left, mid])
            right = max([right, mid])
            # Does all landmarks fit into this box?
            if top < true_bbox[1] or bot > true_bbox[1] + true_bbox[3] or left < true_bbox[0] or right > true_bbox[0] + \
                    true_bbox[2]:
                print('landmarkd outside of bbox: ', img_path)
                outside_landmarks_cnt += 1
                continue
            # 人脸扩充、点、框、原点偏移变化
            cropped_face, cropped_pts, cropped_box, cropped_offset = crop_face(img, true_bbox, true_pts)

            # predict_pts,predict_theta,result_img = predict_landmarks(cropped_face,cropped_offset,
            #                                                output_theta_layer_name,image_size)
            predict_pts, result_img = predict_landmarks(sess,cropped_face,
                                                        cropped_offset,X,pred_pt) #,pred_theta)

            # centers，corners，diagonal
            Normal = 'centers'
            error_all = calculate_NME(true_pts,predict_pts,target_parts=None,normalization=Normal)
            error_eye = calculate_NME(true_pts,predict_pts,target_parts=eye_idx_list,normalization=Normal)
            error_brow = calculate_NME(true_pts,predict_pts,target_parts=eyebrow_idx_list,normalization=Normal)
            error_nose = calculate_NME(true_pts,predict_pts,target_parts=nose_idx_list,normalization=Normal)
            error_mouth = calculate_NME(true_pts,predict_pts,target_parts=mouth_idx_list,normalization=Normal)
            error_contour = calculate_NME(true_pts,predict_pts,target_parts=contour_idx_list,normalization=Normal)
            error_noface = calculate_NME(true_pts,predict_pts,target_parts=nothing_face_idx_list,normalization=Normal)
            error_all_list.append(error_all)
            error_eye_list.append(error_eye)
            error_brow_list.append(error_brow)
            error_nose_list.append(error_nose)
            error_mouth_list.append(error_mouth)
            error_contour_list.append(error_contour)
            error_nothing_face_list.append(error_noface)
            if DEBUG:
                # 原图标注
                draw_img = cropped_face.copy()
                for pt in cropped_pts:
                    cv2.circle(draw_img,(int(pt[0]),int(pt[1])),2,(255,255,0))
                cv2.rectangle(draw_img, (int(cropped_box[0]), int(cropped_box[1])),
                              (int(cropped_box[2] + cropped_box[0]), int(cropped_box[3] + cropped_box[1])),
                              (0, 255, 255), 2)
                cv2.imencode('.jpg', draw_img)[1].tofile(
                    os.path.join(save_dir,'org_pic', '%s_true_crop.jpg' % (os.path.splitext(name)[0])))
                # 原图预测 -- 显示在原图上
                draw_img2 = img.copy()
                for i in range(68):
                    cv2.circle(draw_img2, (int(predict_pts[i][0]), int(predict_pts[i][1])), 1,
                               (0, 0, 255), 2)
                #cv2.imshow('test_pre', draw_img2)
                cv2.imencode('.jpg', draw_img2)[1].tofile(
                    os.path.join(save_dir,'test_result', '%s_predict_src.jpg' % (os.path.splitext(name)[0])))
                #cv2.putText(result_img, 'pitch: %0.3f' % predict_theta[0][0], (60, 60), font, 0.5, (255, 255, 0))
                #cv2.putText(result_img, 'roll:  %0.3f' % predict_theta[0][1], (60, 80), font, 0.5, (255, 0, 255))
                #cv2.putText(result_img, 'yaw:   %0.3f' % predict_theta[0][2], (60, 100), font, 0.5, (0, 255, 255))
                cv2.putText(result_img,'This pic error: %0.3f ' % error_all ,(60,80),font,0.5,(0,0,255))
                #cv2.imshow('Test result',result_img)
                cv2.imwrite('./test_result/test_crop/%s_test_crop.jpg' % name.split('.')[0],result_img)
                # cv2.waitKey(200)
        print('all NME: ', np.mean(error_all_list))
        print('noface NME: ', np.mean(error_nothing_face_list))
        print('contour NME: ', np.mean(error_contour_list))
        print('brow NME: ', np.mean(error_brow_list))
        print('nose NME: ', np.mean(error_nose_list))
        print('eye NME: ', np.mean(error_eye_list))
        print('mouth NME: ', np.mean(error_mouth_list))



if __name__ == '__main__':
    font = cv2.FONT_HERSHEY_SIMPLEX  # font
    DEBUG = True
    model_eval()









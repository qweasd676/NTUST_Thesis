# @time 2021/12/30 1:32 PM
# @file main.py
#

import cv2
import pandas as pd
import numpy as np
import threading
import time
import glob
import os
import csv
from matplotlib import pyplot as plt
from src.Segmentation_Algorithm import *
from src.Corner_Algorithm import *
from src.Wavelet_Transform import *
from src.opencv import *
from src.image_to_video import *
from src.video_to_jpg import *
from src.rename_image  import *
from src.Optimze_result import *


# common sense
# YUV色彩空間是把亮度(luma)與色度(Chroma)分離
# “Y”表示亮度，也就是灰度值。
# “U”表示蓝色通道与亮度的差值。
# “V”表示红色通道与亮度的差值。

def process_video(sel):

    if sel:
        pos = './'
        # pos = './video/calib-charuco/'
        # pos = './locust/MVS_dataset/0613/jump_0617/'
        threads = []
        video_type = '*.avi'
        video_path = sorted(glob.glob( pos + video_type))
        print(video_path)
        
        for i in range(len(video_path)):
            file_path = os.path.splitext(video_path[i])[0]
            file_name = file_path.split('/')[-1]
            count = 1  # count 表示要間隔多少
            threads.append(threading.Thread(target = video_to_pic, args = (video_path[i], pos+file_name+'/','png',count)))
            threads[i].start()
        
        for i in range(len(video_path)):
            threads[i].join()
    else: 
# /home/ntustee_chen/Desktop/image_prcoessing/locust/MVS_dataset/calibration/0620/47124104
        # path = './locust/MVS_dataset/calibration/0620/'
        path = './exp_paper/'
        camA = 'exp_jump'
        camB = 'exp_run'
        # camC = '47124106'
        type_func = 'jump'  # checkerboard or jump or run
        type_img  = 'png'

        pic_to_video(path + camA + '/*.' + type_img, type_func +'-camA.mp4',5)
        pic_to_video(path + camB + '/*.' + type_img, type_func +'-camB.mp4',5)
        # pic_to_video(path + camC + '/*.' + type_img, type_func +'-camC.mp4',30)

    print("Done.")


def two_image_combime(sel):

    if sel:
        path = './locust/combine/'
        img1 = cv2.imread(path + "image_analysis_angle_ffl1.jpg")
        img11 = cv2.imread(path + "image_analysis_angle_mfr1.jpg")
        img2 = cv2.imread(path + "image_analysis_angle_ffr1.jpg")
        img22 = cv2.imread(path + "image_analysis_angle_mfl1.jpg")
        img3 = cv2.imread(path + "image_analysis_angle_hfl1.jpg")
        img33 = cv2.imread(path + "image_analysis_angle_hfr1.jpg")
        
        img_1 = np.hstack((img1,img11))
        img_2 = np.hstack((img2,img22))
        img_3 = np.hstack((img3,img33))


        cv2.imwrite('ffl_mfr.jpg',img_1)
        cv2.imwrite('ffr_mfl.jpg',img_2)
        cv2.imwrite('hfl_hfl.jpg',img_3)



    else:
        # /home/ntustee_chen/Desktop/image_prcoessing/locust/MVS_dataset/0613/run
        path = './locust/MVS_dataset/0613/jump_0617/RGB_2330_2370/47124104/*.jpg'
        path1 = './locust/MVS_dataset/0613/jump_0617/RGB_2330_2370/47124104/'
        path2 = './locust/MVS_dataset/0613/jump_0617/RGB_2330_2370/47124105/'
        path3 = './locust/MVS_dataset/0613/jump_0617/RGB_2330_2370/47124106/'
    # /home/ntustee_chen/Desktop/image_prcoessing/locust/MVS_dataset/0613/jump_0617/RGB_2330_2370/47124104

        frame_list = sorted(glob.glob(path))

        for idx, path_idx in enumerate(frame_list): 
            path_idx = path_idx.rsplit("/",1)
            img1 = cv2.imread(path1 + path_idx[-1])
            img2 = cv2.imread(path2 + path_idx[-1])
            img3 = cv2.imread(path3 + path_idx[-1])

            img = np.hstack((img1,img2,img3))
            cv2.imwrite('./locust/MVS_dataset/0613/jump_0617/combined/{0:03d}.jpg'.format(idx),img)

        # # for x in range(0 , 437):
        #     img1 = cv2.imread(path1 + 'image00{0}.jpg'.format(str(x)))
        #     img2 = cv2.imread(path2 + 'image00{0}.jpg'.format(str(x)))
        #     img3 = cv2.imread(path3 + 'image00{0}.jpg'.format(str(x)))

        #     img = np.hstack((img1,img2,img3))
        #     # print(1)
        #     cv2.imwrite('./locust/MVS_dataset/0613/run/combine/{0:03d}.jpg'.format(x),img)
    return 0

def process():

    # draw2dpoint()
    # draw2dpoint_sp()
    process_video(0)  # select 1 is video to picure, and then select 0 is picture to video.

    path_img = './locust/MVS_dataset/video_MVS_0605/run-camB/*.png'
    path = './locust/MVS_dataset/video_MVS_0605/pha-camB/*.png' 
    
    # path_img = './locust/MVS_dataset/stereo_run/47124106_0309_2/*.jpg'
    # path = './locust/MVS_dataset/stereo_run/pha2/*.png' 

    # path_img = './locust/paper_exp/C0147_feature/*.png'
    # path = './locust/paper_exp/pha_C0147/*.png'



# /home/ntustee_chen/Desktop/image_prcoessing/locust/paper_exp/C0154_feature

    # image_OTSU(path,path_img)
    # image_run(path,path_img,'one_jump_analysis_C0147')
    # image_run(path,path_img,'stereo_run_analysis')
    # image_run(path,path_img,'run_camB_analysis')


    cv2.destroyAllWindows()


def analysis():
    # # analysis 

    # path = '/media/ntustee_chen/Extreme SSD/m10907305/thesis progress/論文實驗圖/D部份/MVS_jump.csv'
    path = '/media/ntustee_chen/Extreme SSD/m10907305/thesis progress/論文實驗圖/D部份/MVS_run.csv'
    # path = '/media/ntustee_chen/Extreme SSD/m10907305/thesis progress/論文實驗圖/D部份/stereo_jump.csv'
    # path = '/media/ntustee_chen/Extreme SSD/m10907305/thesis progress/論文實驗圖/D部份/stereo_run.csv'


    # path = '/media/ntustee_chen/Extreme SSD/m10907305/thesis progress/論文實驗圖/B部份/MVS_run.csv'
    # path = '/media/ntustee_chen/Extreme SSD/m10907305/thesis progress/論文實驗圖/B部份/MVS_jump.csv'

    analyze_3D_points(path)

    # angel_3D_point(path,1)
        
    # find_error_3d(path)

    # path = './csv/result.csv'
    # analysis_list = sorted(glob.glob(path))
    # print(analysis_list)
    # deeplabcut_csv_analysis(analysis_list)

    # regression_ml('d')

    # deal_with_csv(path)
    # analyze_3D_feet(path)

if __name__ == "__main__":

    # process()
    analysis()


    #修改照片名稱
    # two_image_combime(0)
    # rename_function('./locust/paper_exp/C0154_feature/',1,'png')
    # rename_function('./locust/MVS_dataset/0613/run/','47124104','jpg')
    # rename_function('./locust/MVS_dataset/0613/run/','47124105','jpg')
    # rename_function('./locust/MVS_dataset/0613/run/','47124106','jpg')
    # rename_function('./locust/47124105_0309_2/','47124104','jpg')



 



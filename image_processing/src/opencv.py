import numpy as np
import cv2
import os
import pandas as pd
import glob
from src.Corner_Algorithm import *


def Histogram_Equalization_df(img):
    #this is the RGB channel
    # convert the image into grayscale before doing histogram equalization
    img_yuv = cv2.cvtColor(img , cv2.COLOR_BGR2YUV)
    
    # image equalization
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    
    hist = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return hist

def CLAHE_Equalization_df(img):
    #this is the RGB channel
    # convert the image into grayscale before doing histogram equalization
    img_yuv = cv2.cvtColor(img , cv2.COLOR_BGR2YUV)
    
    # image equalization
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    
    # create clahe image
    clahe = cv2.createCLAHE()
    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img_output


def image_OTSU(path,path_img):
        
    img_file_gray = sorted(glob.glob(path))
    img_file_orignal = sorted(glob.glob(path_img))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))


    for index,img in enumerate(img_file_gray):
        img_pha = cv2.imread(img,0)
        # img_pha_3 = cv2.imread(img)
        img_original = cv2.imread(img_file_orignal[index])
        # thresh, mask = cv2.threshold(img_gray, 120, 255, cv2.THRESH_BINARY)
        # thresh, img_pha = cv2.threshold(img_pha, 0, 255, cv2.THRESH_OTSU)
        # mask =  np.concatenate((mask,mask,mask),axis =  -1)

        # 先把原圖跟pha合併轉成灰階


        result = cv2.bitwise_and(img_original, img_original, mask = img_pha)
        result_gray = cv2.cvtColor(result , cv2.COLOR_BGR2GRAY)
        result_sharpen = sharpen(result_gray)

        # #bg主要是白色的，這樣的作法可以把背景塗滿成white
        # mask_bg = result_sharpen == 0
        # result_combine = result_sharpen + ~result_sharpen * mask_bg
        # #先透過sobel和ridge邊緣檢測
        # output = findcorner_algorithm(img_original, result_combine)


        # #將sobel結果進行侵蝕膨脹
        # img_erode = cv2.erode(output.sobel_result,kernel,iterations = 0)
        # img_dilate = cv2.dilate(img_erode,kernel,iterations = 0)
        # # sobel_res = ((img_dilate > 10) *1).astype('uint8')
        # result_morphology = cv2.bitwise_and(img_original, img_original, mask = img_dilate)


        img_erode = cv2.erode(result_sharpen,kernel,iterations = 0)
        img_dilate = cv2.dilate(img_erode,kernel,iterations = 0)
        # sobel_res = ((img_dilate > 10) *1).astype('uint8')
        result_morphology = cv2.bitwise_and(img_original, img_original, mask = img_dilate)     


        result_nonmorphology = cv2.bitwise_and(img_original, img_original, mask = result_sharpen)
        result_bg = result_nonmorphology ==0 
        result_combine_rgb = result_nonmorphology + ~result_nonmorphology * result_bg
        # result_combine_rgb_ = result_combine_rgb.copy()

        #整合灰階與sobel與原圖的結果
        row,col,__ = result_morphology.shape
        row = int(row/10)
        col = int(col/10)

        cv2.putText(result_morphology, "morphology", (col,row), cv2.FONT_HERSHEY_PLAIN,5, (255, 255, 122), 5, cv2.LINE_AA)
        cv2.putText(result_nonmorphology, "nonmorphology", (col,row), cv2.FONT_HERSHEY_PLAIN,5, (255, 255, 122), 5, cv2.LINE_AA)
        cv2.putText(result_combine_rgb, "add white bg",  (col,row), cv2.FONT_HERSHEY_PLAIN,5, (255, 255, 122), 5, cv2.LINE_AA)
        res = np.hstack((result_morphology,result_nonmorphology,result_combine_rgb))
        #灰階化的圖片只有單通道，必須轉換成三通道

        cv2.putText(result_gray, "Gray", (col,row), cv2.FONT_HERSHEY_PLAIN,5, (255, 255, 122), 5, cv2.LINE_AA)
        cv2.putText(result_sharpen, "result_sharpen", (col,row), cv2.FONT_HERSHEY_PLAIN,5, (255, 255, 122), 5, cv2.LINE_AA)
        cv2.putText(img_dilate, "result_sharpen + morphology",  (col,row), cv2.FONT_HERSHEY_PLAIN,5, (255, 255, 122), 5, cv2.LINE_AA)

        print(result_gray.shape,result_morphology.shape,img_dilate.shape)

        res_gray = np.hstack((result_gray,result_morphology[:,:,0],img_dilate))
        res_gray = np.expand_dims(res_gray,axis = 2)
        res_gray = np.concatenate((res_gray,res_gray,res_gray),axis= -1)

        res = np.vstack((res,res_gray))
        res = cv2.resize(res,(1280,720))
        cv2.imshow('result_RGB',res)
        # cv2.imshow('result_gray',result_combine)       
        # cv2.imshow('result_GRAY',cv2.resize(res_gray,(1280,720)))

        # cv2.imshow('Ridge_Detection',~output.ridges)
        cv2.imwrite('./analysis/{0}.jpg'.format(index),result_combine_rgb)
        cv2.waitKey(0)
        if index == 1:
            break

def image_run(path,path_img,result_file):
        
    img_file_gray = sorted(glob.glob(path))
    img_file_orignal = sorted(glob.glob(path_img))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    mkdir_df('./analysis/{0}'.format(result_file))


    for index,img in enumerate(img_file_gray):
        y_l = 50       #0
        y_h = 250          #540
        x_l = 300       #0
        X_h = 600    #720

        img_pha = cv2.imread(img,0)
        img_pha = cv2.resize(img_pha, [1440,1080])
        img_pha  = img_pha[y_l*2:y_h*2,x_l*2:X_h*2]

        img_original = cv2.imread(img_file_orignal[index])
        img_original = cv2.resize(img_original, [1440,1080])
        img_original  = img_original[y_l*2:y_h*2,x_l*2:X_h*2,:]
        
        # cv2.imshow('img_original',img_original)

        # 先把原圖跟pha合併轉成灰階
        result = cv2.bitwise_and(img_original, img_original, mask = img_pha)
        result_gray = cv2.cvtColor(result , cv2.COLOR_BGR2GRAY)
        result_sharpen = sharpen(result_gray)


        img_erode  = cv2.erode(result_sharpen,kernel,iterations = 1)
        img_dilate = cv2.dilate(img_erode,kernel,iterations = 5)


        # img_dilate_bool = ((img_dilate > 10 )*(img_dilate < 240 ) *1).astype('uint8')  #C0147
        # img_dilate_bool = (img_dilate > 100 *1).astype('uint8')   #stereo_run
        img_dilate_bool = (img_dilate > 1 *1).astype('uint8')   #video_MVS_0605/run-camB


        # img_erode  = cv2.erode(img_dilate,kernel,iterations = 0)

        result_morphology = cv2.bitwise_and(img_original, img_original , mask = img_dilate_bool)   

        # cv2.imshow('result_morphology',img_dilate)

        res_gray = np.hstack((img_pha,result_gray,result_sharpen,img_dilate))
        res_gray = np.expand_dims(res_gray,axis = 2)
        res_gray = np.concatenate((res_gray,res_gray,res_gray),axis= -1)
        # res_gray = np.hstack((result_gray,result_sharpen,img_dilate))

        res = np.hstack((img_original,res_gray,result_morphology))

        cv2.imwrite('./analysis/{0}/res{1:08d}.png'.format(result_file,index),cv2.resize(res,[8640,1080]))  
        # cv2.imwrite('./analysis/{0}/{1:08d}.png'.format(result_file,index),result_morphology)


        cv2.waitKey(0)
        if index == 100:
            break

    



def sharpen(img , sigma = 10):
    #src: https://www.wongwonggoods.com/python/python_opencv/opencv-sharpen-images/
    blur_img = cv2.GaussianBlur(img, (0,0), sigma)
    usm = cv2.addWeighted(img, 1.5 , blur_img, -0.5, 0)
    return usm


def mkdir_df(pos):
    if os.path.isdir(pos):
        print("Delete old result folder: {}".format(pos))
        os.system("rm -rf {}".format(pos))

    os.system("mkdir {}".format(pos))
    print('create folder:{}'.format(pos))

def output_Equalization_image(sel,path,target):
    img_file = os.listdir(path)
    #create the file.
    mkdir_df(target)
    #select the Equalization function.
    cnt = 0
    for img in sorted(img_file):
        # print(path+img)
        if cnt % 1 == 0:
            if sel == 1 :
                result_img = Histogram_Equalization_df( cv2.imread(path+img) )
            else:
                result_img = CLAHE_Equalization_df( cv2.imread(path+img) )
            cv2.imwrite(target+img,result_img)
            print(img)
            
        cnt += 1
        # if cnt > 500: 
        #     break
    print('total of picture:{0}'.format(cnt))

def find_contours(img,edge_detection):
    print(edge_detection.shape)

    # edge_detection = cv2.blur(~edge_detection,(3,3))
    
    # edge_detection = cv2.blur(edge_detection,(3,3))
    __,edge_detection = cv2.threshold(edge_detection,100,255,cv2.THRESH_BINARY)
    # edge_detection = edge_detection.astype('int8')


    # cv2.imshow('edge_detection',edge_detection)
    contours, hierarchy = cv2.findContours(edge_detection, cv2.RETR_TREE  ,cv2.CHAIN_APPROX_NONE)  
    counts = 0
    # cv2.drawContours(img, contours , -1 , (255,0,255),3)


    for j in contours:
        area = cv2.contourArea(j)
        if(area > 10000):
            
            x,y,w,h = cv2.boundingRect(j)
            print(w,x,y,h)
            cv2.rectangle(img, (x,y), (x+w,y+h), (153,153,0), 5)
            cv2.drawContours(img, j , -1 , (255,0,255),1)
            # counts += 1         
            # # # calculate moments of binary image
            M = cv2.moments(j)
            # # # calculate x,y coordinate of center
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # # self.object_center_list.append(point(cX,cY))
            # # # put text and highlight the center
            # cv2.circle(img, (cX, cY), 5, (255, 255, 255), -1)
            cv2.putText(img, "Locust", (x , y),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        # display the image
    cv2.imwrite('contours.png',img)
    return img

def drawpoint(path_csv,path,pic_path):
    mkdir_df(pic_path)
    pd_csv = pd.read_csv(path_csv)
    img_file = os.listdir(path)
    size = len(pd_csv.iloc[4,:].values)
    cnt = 3
    # print(pd_csv.iloc[4,:].values)
    for img in sorted(img_file):
        img_ = cv2.imread(path+img)
        for point in range(1,size,3):
            if( np.isnan(float(pd_csv.iloc[cnt,point])) ):
                continue
            if(float(pd_csv.iloc[cnt,point+2]) < 0.8):
                continue
            # print(pd_csv.iloc[cnt,point])
            # print(pd_csv.iloc[cnt,point+1])
            x = float(pd_csv.iloc[cnt,point])
            y = float(pd_csv.iloc[cnt,point+1])
            cv2.circle(img_,(int(x),int(y)),1,(0,0,255),10)
        cv2.imwrite(pic_path+img,img_)
        # break
        cnt +=1

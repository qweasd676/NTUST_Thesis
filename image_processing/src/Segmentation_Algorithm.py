# -*- coding: utf-8 -*-
#
# @file Segmentation_Algorithm.py  
#


import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time
# from skimage.restoration import (denoise_wavelet,estimate_sigma)
# from skimage.util import random_noise
# from skimage.metrics import peak_signal_noise_ratio
# import skimage.io
# from skimage.restoration import (denoise_wavelet, estimate_sigma)
# from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.segmentation import chan_vese,clear_border
from skimage import data, img_as_float
from skimage.segmentation import chan_vese
import pandas as pd
import collections


#region
class segmentation_algorithm:

    def __init__(self,img):
        self.img = img

    def ROI_select(self):
        showCrosshair = False
        fromCenter = False
        r = cv.selectROI("Image",self.img, fromCenter, showCrosshair)
        self.ROI_img = self.img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
        self.ROI_img_dims = np.expand_dims(self.ROI_img,axis = 2)
        self.imCrop = np.concatenate((self.ROI_img_dims,self.ROI_img_dims,self.ROI_img_dims),axis= -1)
        # print(self.ROI_img.shape)

    def watershed_algorithm(self):
        # input image must be gray or binary.
        water_result = self.imCrop.copy()
        # noise removal
        kernel = np.ones((3,3),np.uint8)
        opening = cv.morphologyEx(self.CV_binary,cv.MORPH_OPEN,kernel, iterations = 0)
        # sure background area
        sure_bg = cv.dilate(opening,kernel,iterations=0)
        # cv.imshow('dsadas',sure_bg)
        # Finding sure foreground area
        dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
        cv.imwrite('dist_transform.jpg',dist_transform)
        ret, sure_fg = cv.threshold(dist_transform,0.5*dist_transform.max(),255,0)
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv.subtract(sure_bg,sure_fg)

        # Marker labelling
        ret, markers = cv.connectedComponents(sure_fg)
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers+1
        # Now, mark the region of unknown with zero
        markers[unknown==255] = 0
        self.watershed_connect = markers
        df = pd.DataFrame(markers)
        df.to_csv('watershed.csv',index=False,header=None)
        self.center_point()
        markers = cv.watershed(water_result,markers)

        # df = pd.DataFrame(markers)
        # df.to_csv('watershed.csv',index=False,header=None)
        water_result[markers == -1] = [0,255,0]
        self.watershed_result = water_result
        self.watershed_graphy = markers

    def center_point(self):
        #First, preprocess the matrix of watershed algorithm.
        graph = self.watershed_connect.copy()
        img_center = self.ROI_img.copy()
        df = pd.DataFrame(graph)
        df.to_csv('graph.csv',index=False,header=None)
        graph[graph < 2] = 0
        graph = clear_border((graph*255).astype('uint8'))
        self.center = graph

        #Find the center point of each object, that stored in the deque.
        self.object_center_list = collections.deque()
        point = collections.namedtuple("coordiate",["x","y"])
        counts = 0
        contours, hierarchy = cv.findContours(graph, cv.RETR_TREE  ,cv.CHAIN_APPROX_NONE)  
        for j in contours:
            area = cv.contourArea(j)
            if(area > 0):
                counts += 1         
                # calculate moments of binary image
                M = cv.moments(j)
                # calculate x,y coordinate of center
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                self.object_center_list.append(point(cX,cY))
                # put text and highlight the center
                cv.circle(img_center, (cX, cY), 5, (255, 255, 255), -1)
                cv.putText(img_center, "centroid{0}".format(counts), (cX - 25, cY - 25),cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            # display the image
        cv.imshow("img_center", img_center)
        cv.imwrite("img_center.jpg", img_center)


    def chan_vese_algorithm(self):
        interation_cv = 50
        image = img_as_float(self.ROI_img)
        # start = time.time()
        self.cv_result = chan_vese(image, mu=0.25, lambda1=1, lambda2=1, tol=1e-3, max_iter=interation_cv,
                        dt=0.5, init_level_set="checkerboard", extended_output=True)
        # end = time.time()

        cv_condtion = np.argmax(self.cv_result[2])
        if( cv_condtion < interation_cv*0.3 ):
            self.CV_binary = np.uint8(self.cv_result[0]*255)
        else:
            self.CV_binary = np.uint8(np.abs(self.cv_result[0]-1)*255)
        # print('time respones : {0}'.format(end-start))
  
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        ax = axes.flatten()

        ax[0].imshow(image, cmap="gray")
        ax[0].set_axis_off()
        ax[0].set_title("Original Image", fontsize=12)

        ax[1].imshow(self.cv_result[0], cmap="gray")
        ax[1].set_axis_off()
        title = "Chan-Vese segmentation - {} iterations".format(len(self.cv_result[2]))
        ax[1].set_title(title, fontsize=12)

        ax[2].imshow(self.cv_result[1], cmap="gray")
        ax[2].set_axis_off()
        ax[2].set_title("Final Level Set", fontsize=12)

        ax[3].plot(self.cv_result[2])
        ax[3].set_title("Evolution of energy over iterations", fontsize=12)

        fig.tight_layout()
        plt.show()


#endregion




         


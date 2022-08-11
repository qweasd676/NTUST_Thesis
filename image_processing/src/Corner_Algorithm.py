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
# from skimage import data, img_as_float
# from skimage.segmentation import chan_vese
import pandas as pd
from past.builtins import xrange
from skimage.feature import hog
from skimage import data, exposure


class findcorner_algorithm:

    def __init__(self,img,ROI):
        self.img =img   #gray_img have three channel.
        self.ROI = ROI  #gray_img have one channel.
        self.kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
        # self.SubPixel_Harris()
        self.sobel_detection()
        self.Ridge_Detection()

    def Morphology_img(self):
        self.ROI = cv.erode(self.ROI ,self.kernel, iterations = 5)
        self.ROI = cv.dilate(self.ROI ,self.kernel, iterations = 5)


    def Harris(self):
        # input must be gray, and image have three channel.
        gray_3D_image = self.img.copy()
        gray  = np.float32(self.ROI)
        dst = cv.cornerHarris(gray,2,3,0.04)
        #result is dilated for marking the corners, not important
        dst = cv.dilate(dst,None)
        # Threshold for an optimal value, it may vary depending on the image.
        gray_3D_image[dst>0.01*dst.max()]=[0,0,255]
        self.harris_img = gray_3D_image

    def SubPixel_Harris(self):
        # input must be gray, and image have three channel.
        gray_3D_image = self.img.copy()
        # __, ROI_img = cv.threshold(self.ROI,50,255,0)
        gray = np.float32(self.ROI)  #self.ROI or ROI_img
        # cv.imshow('dsa',ROI_img)
        dst = cv.cornerHarris(gray,2,5,0.04)
        dst = cv.dilate(dst,None)
        ret, dst = cv.threshold(dst,0.01*dst.max(),255,0)
        dst = np.uint8(dst)
        # find centroids
        ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)

        # define the criteria to stop and refine the corners
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

        # Now draw them
        res = np.hstack((centroids,corners))
        res = np.int0(res)
        try:
            gray_3D_image[res[:,1],res[:,0]]=[0,0,255]
            gray_3D_image[res[:,3],res[:,2]] = [0,255,0]
            self.subpixel_harris = gray_3D_image
            self.subpixel_res = res[:,2:]
        except:
            self.subpixel_harris = gray_3D_image
            self.subpixel_res = res[:,2:]
            
    def Ridge_Detection(self):
        ridge_filter = cv.ximgproc.RidgeDetectionFilter_create(ddepth =cv.CV_32FC1, dx = 1,dy = 1,
            ksize = 3,out_dtype=cv.CV_16SC1 ,scale =1 ,delta=0,borderType = cv.BORDER_DEFAULT)
        self.ridges = ridge_filter.getRidgeFilteredImage(self.ROI)
        self.rigdes_int = self.ridges.astype("int8")

        cv.imwrite('ridges.jpg',self.ridges)
        # df = pd.DataFrame(self.ridges )
        # df.to_csv('ridges.csv',index=False,header=None)

    def sobel_detection(self):
        x = cv.Sobel(self.ROI,cv.CV_16S,1,0,ksize = 3)
        y = cv.Sobel(self.ROI,cv.CV_16S,0,1,ksize = 3 )
        absX = cv.convertScaleAbs(x)   # 转回uint8
        absY = cv.convertScaleAbs(y)
        self.sobel_result = cv.addWeighted(absX,0.5,absY,0.5,0)
        # self.sobel_result = cv.addWeighted(x,0.5,y,0.5,0)
        cv.imwrite('sobel_result.jpg',self.sobel_result)
        # df = pd.DataFrame(self.sobel_result )
        # df.to_csv('sobel_result.csv',index=False,header=None)
         

    def Computer_HOG(self):
        fd, hog_image = hog(self.img, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

        ax1.axis('off')
        ax1.imshow(self.img, cmap=plt.cm.gray)
        ax1.set_title('Input image')

        # Rescale histogram for better display
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        # print(hog_image_rescaled.shape)
        df = pd.DataFrame(hog_image_rescaled)
        df.to_csv('hog_image_rescaled.csv',index=False,header=None)
        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')
        plt.show()

    def image_phase(self):
        sobelx = cv.Sobel(self.ROI,cv.CV_32F,1,0,ksize=3)
        sobely= cv.Sobel(self.ROI,cv.CV_32F,0,1,ksize=3)
        phase=cv.phase(sobelx,sobely,angleInDegrees=True)
        # print(type(phase))
        df = pd.DataFrame(phase)
        df.to_csv('phase.csv',index=False,header=None)



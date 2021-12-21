# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 17:12:06 2021

@author: leo90
"""
import math
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import cv2
import GUI as ui
import numpy as np
import matplotlib.pyplot as plt


class Main(QMainWindow, ui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pushButton_1.clicked.connect(self.show_photo)
        self.pushButton_2.clicked.connect(self.colorSeparation)
        self.pushButton_3.clicked.connect(self.colorTransformation)
        self.pushButton_4.clicked.connect(self.BlendingWindow)
        self.pushButton_5.clicked.connect(self.GaussianBlur)
        self.pushButton_6.clicked.connect(self.BilateralFilter)
        self.pushButton_7.clicked.connect(self.MedianFilter)
        self.pushButton_8.clicked.connect(self.GaussianBlur_manual)
        self.pushButton_9.clicked.connect(self.Sobelx)
        self.pushButton_10.clicked.connect(self.Sobely)
        self.pushButton_11.clicked.connect(self.Magnitude)
        self.pushButton_12.clicked.connect(self.Resize)
        self.pushButton_13.clicked.connect(self.Translation)
        self.pushButton_14.clicked.connect(self.RotateScale)
        self.pushButton_15.clicked.connect(self.Shearing)

    def show_image(self, name, img):
        cv2.imshow(name, img)

    def show_photo(self):  # only for Q1
        img = cv2.imread('Sun.jpg')
        self.show_image('Sun', img)

    def colorSeparation(self):
        img = cv2.imread('Sun.jpg')
        (B, G, R) = cv2.split(img)
        patch = np.zeros(img.shape[:2], "uint8")  # unsigned int 8bit
        self.show_image("Red", cv2.merge([patch, patch, R]))
        self.show_image("Green", cv2.merge([patch, G, patch]))
        self.show_image("Blue", cv2.merge([B, patch, patch]))

    def colorTransformation(self):
        img = cv2.imread('Sun.jpg')
        (B, G, R) = cv2.split(img)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height = img.shape[0]
        width = img.shape[1]
        average_weighted_img = np.zeros([height, width], np.uint8)
        for i in range(height):
            for j in range(width):
                (B, G, R) = img[i][j]
                result = (int(B)+int(G)+int(R))/3
                average_weighted_img[i, j] = np.uint8(result)

        self.show_image("grayScale", gray_img)
        self.show_image("average_weigthed", average_weighted_img)

    def updating(self, x):
        alpha = x/255
        beta = 1-alpha
        img_add = cv2.addWeighted(
            self.dog_weak, alpha,  self.dog_strong, beta, self.gamma)
        self.show_image('image', img_add)

    def BlendingWindow(self):
        self.dog_weak = cv2.imread('Dog_Weak.jpg')
        self.dog_strong = cv2.imread('Dog_strong.jpg')
        alpha = 0
        beta = 1-alpha
        self.gamma = 0
        img_add = cv2.addWeighted(
            self.dog_weak, alpha,  self.dog_strong, beta, self.gamma)
        cv2.namedWindow('image')
        self.show_image('image', img_add)
        cv2.createTrackbar('Blend', 'image', 0, 255, self.updating)

    def GaussianBlur(self):
        img = cv2.imread('Lenna_whiteNoise.jpg')
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        self.show_image('Gaussian Blur', blur)
        self.show_image('Original', img)

    def BilateralFilter(self):
        img = cv2.imread('Lenna_whiteNoise.jpg')
        Bilateral = cv2.bilateralFilter(img, 9, 90, 90)
        self.show_image('BilateralFilter', Bilateral)
        self.show_image('Original', img)

    def MedianFilter(self):
        img = cv2.imread('Lenna_ pepperSalt.jpg')
        median_5 = cv2.medianBlur(img, 5)
        median_3 = cv2.medianBlur(img, 3)
        self.show_image('MedianFilter_3x3', median_3)
        self.show_image('MedianFilter_5x5', median_5)
        self.show_image('Original', img)

    def convolution(self, image, kernel, average=False):
        image_row, image_col = image.shape
        kernel_row, kernel_col = kernel.shape
        output = np.zeros(image.shape)
        
        padded_image = cv2.copyMakeBorder(
            image,
            top=1,
            bottom=1,
            left=1,
            right=1,
            borderType=cv2.BORDER_DEFAULT,
        )
        
        for row in range(image_row):
            for col in range(image_col):
                output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
                
                if average:
                    output[row, col] /= kernel.shape[0] * kernel.shape[1]
        # print(type(output[0][0]))
        # print("Output Image size : {}".format(output.shape))        
        return output

    def GaussianBlur_manual(self):
        img = cv2.imread('House.jpg')
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        y, x = np.mgrid[-1:1:3j, -1:1:3j]
        gaussian_kernel = np.exp(-(x**2+y**2))
        gaussian_kernel = gaussian_kernel/gaussian_kernel.sum()
        # gaussianblur_manual = cv2.filter2D(gray_img,-1,gaussian_kernel,anchor=(-1, -1),delta=0)

        gaussianblur_manual = self.convolution(
            gray_img, gaussian_kernel, average=False)
        gaussianblur_manual = np.array(gaussianblur_manual, 'uint8')
        self.show_image('GaussianBlur', gaussianblur_manual)
        self.show_image('Original', gray_img)

    def Sobelx(self):
        img = cv2.imread('House.jpg')
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   
        y, x = np.mgrid[-1:1:3j, -1:1:3j]
        gaussian_kernel = np.exp(-(x**2+y**2))
        gaussian_kernel = gaussian_kernel/gaussian_kernel.sum()
        gaussianblur_manual = self.convolution(gray_img, gaussian_kernel)
        gaussianblur_manual = np.array(gaussianblur_manual, 'uint8')
        x1 = np.mgrid[-1:1:3j]
        y1 = np.mgrid[-2:2:3j]
        sobelx_kernel = np.vstack((x1, y1, x1))
        sobelx_kernel = np.array(sobelx_kernel,'float32')
        gaussianblur_manual_float32 = np.array(gaussianblur_manual,'float32')
        sobelx_manual = self.convolution(gaussianblur_manual_float32, sobelx_kernel)     
        sobelx_manual_row = sobelx_manual.shape[0]
        sobelx_manual_col = sobelx_manual.shape[1]
        
        
        for row in range(sobelx_manual_row):
            for col in range(sobelx_manual_col):
                sobelx_manual[row][col] = abs(sobelx_manual[row][col])
        sobelx_manual = np.array(sobelx_manual,'uint8')
        
        
        
        # print(sobelx_manual[0])
        self.show_image('GaussianBlur', gaussianblur_manual)
        self.show_image('Sobelx', sobelx_manual)
        # self.show_image('Sobelx_cv', sobelx_manual_cv)
       

    def Sobely(self):
        img = cv2.imread('House.jpg')
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        y, x = np.mgrid[-1:1:3j, -1:1:3j]
        gaussian_kernel = np.exp(-(x**2+y**2))
        gaussian_kernel = gaussian_kernel/gaussian_kernel.sum()
        gaussianblur_manual = self.convolution(gray_img, gaussian_kernel)
        gaussianblur_manual = np.array(gaussianblur_manual,'uint8')
        x1 = np.mgrid[-1:1:3j]
        y1 = np.mgrid[-2:2:3j]
        sobely_kernel = np.vstack((x1, y1, x1))
        sobely_kernel = sobely_kernel.transpose()
        sobely_manual = self.convolution(
            gaussianblur_manual, sobely_kernel, average=False)
        sobely_manual_row = sobely_manual.shape[0]
        sobely_manual_col = sobely_manual.shape[1]
        for row in range(sobely_manual_row):
            for col in range(sobely_manual_col):
                sobely_manual[row][col] = abs(sobely_manual[row][col])
        sobely_manual = np.array(sobely_manual,'uint8')
        self.show_image('GaussianBlur', gaussianblur_manual)
        self.show_image('Sobely', sobely_manual)

    def Magnitude(self):
        img = cv2.imread('House.jpg')
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        y, x = np.mgrid[-1:1:3j, -1:1:3j]
        gaussian_kernel = np.exp(-(x**2+y**2))
        gaussian_kernel = gaussian_kernel/gaussian_kernel.sum()
        gaussianblur_manual = self.convolution(gray_img, gaussian_kernel)
        gaussianblur_manual = np.array(gaussianblur_manual, 'uint8')
        x1 = np.mgrid[-1:1:3j]
        y1 = np.mgrid[-2:2:3j]
        sobelx_kernel = np.vstack((x1, y1, x1))
        sobelx_manual = self.convolution(
            gaussianblur_manual, sobelx_kernel, average=False)
        sobelx_manual_row = sobelx_manual.shape[0]
        sobelx_manual_col = sobelx_manual.shape[1]
        for row in range(sobelx_manual_row):
            for col in range(sobelx_manual_col):
                sobelx_manual[row][col] = abs(sobelx_manual[row][col])
        sobelx_manual = np.array(sobelx_manual, 'uint8')
        
        sobely_kernel = sobelx_kernel.transpose()

        sobely_manual = self.convolution(
            gaussianblur_manual, sobely_kernel, average=False)
        sobely_manual_row = sobely_manual.shape[0]
        sobely_manual_col = sobely_manual.shape[1]
        for row in range(sobely_manual_row):
            for col in range(sobely_manual_col):
                sobely_manual[row][col] = abs(sobely_manual[row][col])
        sobely_manual = np.array(sobely_manual, 'uint8')
        
        height = sobelx_manual.shape[0]
        width = sobelx_manual.shape[1]

        magnitude = np.zeros([height, width], np.uint8)
        for i in range(height):
            for j in range(width):
                result = (sobely_manual[i][j]**2+sobelx_manual[i][j]**2)**0.5
                magnitude[i, j] = np.uint8(result)
        self.show_image('Magnitude', magnitude)
        self.show_image('Sobely', sobely_manual)
        self.show_image('Sobelx', sobelx_manual)

    def Resize(self):
        img = cv2.imread('SQUARE-01.png')
        img_resize = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        # self.show_image('Original', img)
        self.show_image('Resize', img_resize)

    def Translation(self):
        img = cv2.imread('SQUARE-01.png')
        img_resize = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        M = np.float32([[1, 0, 0], [0, 1, 60]])
        translation = cv2.warpAffine(img_resize, M, (400, 300))
        self.show_image('Translation', translation)

    def RotateScale(self):
        img = cv2.imread('SQUARE-01.png')
        img_resize = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        M = np.float32([[1, 0, 0], [0, 1, 60]])
        translation = cv2.warpAffine(img_resize, M, (400, 300))
        M = cv2.getRotationMatrix2D(
            (translation.shape[0]/2, translation.shape[1]/2), 10, 0.5)
        rotateScale = cv2.warpAffine(translation, M, (400, 300))
        self.show_image('Rotate, Scale', rotateScale)

    def Shearing(self):
        img = cv2.imread('SQUARE-01.png')
        img_resize = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        M = np.float32([[1, 0, 0], [0, 1, 60]])
        translation = cv2.warpAffine(img_resize, M, (400, 300))
        M = cv2.getRotationMatrix2D(
            (translation.shape[0]/2, translation.shape[1]/2), 10, 0.5)
        rotateScale = cv2.warpAffine(translation, M, (400, 300))
        pos1 = np.float32([[50, 50], [200, 50], [50, 200]])
        pos2 = np.float32([[10, 100], [200, 50], [100, 250]])
        M = cv2.getAffineTransform(pos1, pos2)
        shearing = cv2.warpAffine(rotateScale, M, (400, 300))
        self.show_image('Shearing', shearing)


# main
if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec_())

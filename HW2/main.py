# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 01:08:45 2021

@author: leo90
"""
import math
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import cv2 as cv
import GUI as ui
import matplotlib.pyplot as plt
import glob

class Main(QMainWindow, ui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.Find_corner)
        self.pushButton_2.clicked.connect(self.Intrinsic)
        self.pushButton_3.clicked.connect(self.Extrinsic)
        self.pushButton_4.clicked.connect(self.Distortion)
        self.pushButton_5.clicked.connect(self.Show_result)
        
    #start from the Q2.1
    def Find_corner(self):
        # termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp=np.zeros((8*11, 3), np.float32)
        objp[:, :2]=np.mgrid[0:8, 0:11].T.reshape(-1, 2)
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        images = glob.glob('*.bmp')
        for fname in images:
            
            img = cv.imread(fname)
            img_resize = cv.resize(img, (1300, 1300), interpolation=cv.INTER_AREA)
            gray = cv.cvtColor(img_resize, cv.COLOR_BGR2GRAY)
            
            # Find the chess board corners
            ret, corners=cv.findChessboardCorners(gray, (8,11), None)
            # If found, add object points, image points (after refining them)
            if ret == True:
                # print(fname)
                corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                objpoints.append(objp)
                imgpoints.append(corners)
                corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                # Draw and display the corners
            cv.drawChessboardCorners(img_resize, (8,11), corners2, ret)
            cv.imshow('img', img_resize)
            cv.waitKey(500)
        cv.destroyAllWindows()
    
    def Intrinsic(self):
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp=np.zeros((8*11, 3), np.float32)
        objp[:, :2]=np.mgrid[0:8, 0:11].T.reshape(-1, 2)
        #store the world coord. and image coord. points
        objpoints=[]
        imgpoints=[]
        #images save in folder "hw6"
        images=glob.glob('*.bmp') 
    
        for fname in images:
            img=cv.imread(fname)
            gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            #find the corner of checkboard
            ret, corners=cv.findChessboardCorners(gray, (8,11), None)
            #save the points as long as find enough pair points
            if ret==True:
                cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                objpoints.append(objp)
                imgpoints.append(corners)
    
        #calibration
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        print("Intrinsic : ")
        print(mtx)
        
    
    def Extrinsic(self):
        index = self.lineEdit.text()
        index = int(index)
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        objp=np.zeros((8*11, 3), np.float32)
        objp[:, :2]=np.mgrid[0:8, 0:11].T.reshape(-1, 2)
        #store the world coord. and image coord. points
        objpoints=[]
        imgpoints=[]
        #images save in folder "hw6"
        images=glob.glob('*.bmp') 
        for fname in images:
            img=cv.imread(fname)
            gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            #find the corner of checkboard
            ret, corners=cv.findChessboardCorners(gray, (8,11), None)
            #save the points as long as find enough pair points
            if ret==True:
                cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                objpoints.append(objp)
                imgpoints.append(corners)
        #calibration
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        rotation_matrix = cv.Rodrigues(rvecs[index],jacobian=0)
        rotation_matrix = np.array(rotation_matrix[0])
        Extrinsic = np.concatenate((rotation_matrix,tvecs[index]), axis = 1)
        print("Extrinsic : ")
        print(Extrinsic)
    
    def Distortion(self):
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp=np.zeros((8*11, 3), np.float32)
        objp[:, :2]=np.mgrid[0:8, 0:11].T.reshape(-1, 2)
        #store the world coord. and image coord. points
        objpoints=[]
        imgpoints=[]
        #images save in folder "hw6"
        images=glob.glob('*.bmp') 
    
        for fname in images:
            img=cv.imread(fname)
            gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            #find the corner of checkboard
            ret, corners=cv.findChessboardCorners(gray, (8,11), None)
            #save the points as long as find enough pair points
            if ret==True:
                cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                objpoints.append(objp)
                imgpoints.append(corners)
        #calibration
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        print("distortion : ")
        print(dist)
    
    def Show_result(self):
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp=np.zeros((8*11, 3), np.float32)
        objp[:, :2]=np.mgrid[0:8, 0:11].T.reshape(-1, 2)
        #store the world coord. and image coord. points
        objpoints=[]
        imgpoints=[]
        #images save in folder "hw6"
        images=glob.glob('*.bmp') 
        for fname in images:
            img=cv.imread(fname)
            img_resize = cv.resize(img, (800, 800), interpolation=cv.INTER_AREA)
            gray=cv.cvtColor(img_resize, cv.COLOR_BGR2GRAY)
            #find the corner of checkboard
            ret, corners=cv.findChessboardCorners(gray, (8,11), None)
            #save the points as long as find enough pair points
            if ret==True:
                cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                objpoints.append(objp)
                imgpoints.append(corners)
        #calibration
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        for fname in images:
            img_test = cv.imread(fname)
            img_test = cv.resize(img_test, (815, 815), interpolation=cv.INTER_AREA)
            img_original = cv.imread(fname)
            img_original = cv.resize(img_original, (800,800), interpolation=cv.INTER_AREA)
            h,  w = img_test.shape[:2]
            newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
            img_result = cv.undistort(img_test, mtx, dist, None, newcameramtx)
            x, y, w, h = roi
            img_result = img_result[y:y+h, x:x+w]
            Hori = np.concatenate((img_result, img_original), axis=1)
            cv.imshow("undistort", Hori)
            cv.waitKey(500)
        
if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec_())
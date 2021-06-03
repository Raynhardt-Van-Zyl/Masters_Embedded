import numpy as np
import cv2
import os
import time

import rospy
from sensor_msgs.msg import Image

from collections import defaultdict, namedtuple

from threading import Thread, Lock
from multiprocessing import Process, Queue

from cv_bridge import CvBridge



class ImageReader(object):

    def __init__(self, cam=None, side=None):

        rospy.Subscriber("/camera/" + side, Image, self.getImage, queue_size=1)

        self.timestamp = None
        self.cam = cam
        self.cache = dict()
        self.idx = 0
        self.image = None

        self.ahead = 10      # 10 images ahead of current index
        self.waiting = 1.5   # waiting time

        self.thread_started = False

        self.bridge = CvBridge()

    def getImage(self,data):
        rospy.loginfo("Received")
        self.image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        self.timestamp = rospy.Time.now()

    def read(self, path):
        img = cv2.imread(self.image, -1)
        if self.cam is None:
            return img
        else:
            return self.cam.rectify(img)

    @property
    def dtype(self):
        return self[0].dtype
    @property
    def shape(self):
        return self.image.shape

class Stereo(object):
    def __init__(self):
        Cam = namedtuple('cam', 'fx fy cx cy width height baseline')
        self.cam = Cam(718.856, 718.856, 607.1928, 185.2157, 1241, 376, 0.5371657)
        self.left = ImageReader(self.cam,"left")
        self.right = ImageReader(self.cam,"right")
        self.timestamp = self.left.timestamp

class Camera(object):
    def __init__(self, 
            width, height,
            intrinsic_matrix, 
            undistort_rectify=False,
            extrinsic_matrix=None,
            distortion_coeffs=None,
            rectification_matrix=None,
            projection_matrix=None):

        self.width = width
        self.height = height
        self.intrinsic_matrix = intrinsic_matrix
        self.extrinsic_matrix = extrinsic_matrix
        self.distortion_coeffs = distortion_coeffs
        self.rectification_matrix = rectification_matrix
        self.projection_matrix = projection_matrix
        self.undistort_rectify = undistort_rectify
        self.fx = intrinsic_matrix[0, 0]
        self.fy = intrinsic_matrix[1, 1]
        self.cx = intrinsic_matrix[0, 2]
        self.cy = intrinsic_matrix[1, 2]

        if undistort_rectify:
            self.remap = cv2.initUndistortRectifyMap(
                cameraMatrix=self.intrinsic_matrix,
                distCoeffs=self.distortion_coeffs,
                R=self.rectification_matrix,
                newCameraMatrix=self.projection_matrix,
                size=(width, height),
                m1type=cv2.CV_8U)
        else:
            self.remap = None

    def rectify(self, img):
        if self.remap is None:
            return img
        else:
            return cv2.remap(img, *self.remap, cv2.INTER_LINEAR)

class StereoCamera(object):
    def __init__(self, left_cam, right_cam):
        self.left_cam = left_cam
        self.right_cam = right_cam

        self.width = left_cam.width
        self.height = left_cam.height
        self.intrinsic_matrix = left_cam.intrinsic_matrix
        self.extrinsic_matrix = left_cam.extrinsic_matrix
        self.fx = left_cam.fx
        self.fy = left_cam.fy
        self.cx = left_cam.cx
        self.cy = left_cam.cy
        self.baseline = abs(right_cam.projection_matrix[0, 3] / 
            right_cam.projection_matrix[0, 0])
        self.focal_baseline = self.fx * self.baseline

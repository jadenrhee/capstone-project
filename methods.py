import cv2
import pyzed.sl as sl
import time
import mediapipe as mp
import math

def getDepth(zedCam, pointCloud, x = None, y = None):
    # Retrieve the point cloud
    imageSize = zedCam.get_camera_information().camera_configuration.resolution
    zedCam.retrieve_measure(pointCloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, imageSize)
    if x == None or y == None: #center of the image
        x = imageSize.width // 2
        y = imageSize.height // 2
    err, pointCloudValue = pointCloud.get_value(x, y)
    if err == sl.ERROR_CODE.SUCCESS:
        return f"Depth at ({x}, {y}): {pointCloudValue[2]} meters"

def isEven(num):
    return True if num % 2 == 0 else False

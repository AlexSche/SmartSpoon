import cv2 
import depthai 
import numpy as np
import cvlib as cv
from cvlib.object_detection import draw_bbox

pipeline = depthai.Pipeline()
cam = pipeline.create(depthai.node.ColorCamera)
cam.setColorOrder(depthai.ColorCameraProperties.ColorOrder.RGB)
cam.setPreviewSize(1280,720)
xout = pipeline.create(depthai.node.XLinkOut)
xout.setStreamName("rgb")
cam.preview.link(xout.input)

object_detector = cv2.createBackgroundSubtractorMOG2()


# Connect to device and start pipeline
with depthai.Device(pipeline) as device:
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False) # type: ignore
    while True:
        #this frame is displayed in BGR
        BGRframe = qRgb.get().getFrame()
        #set frame to RGB
        RGBframe = cv2.cvtColor(BGRframe, cv2.COLOR_RGB2BGR)

        #---  cvlib common object detection
        #bbox, label, conf = cv.detect_common_objects(RGBframe)
        #output_image = draw_bbox(RGBframe, bbox, label, conf)
        # ---

        #Object detection
        mask = object_detector.apply(RGBframe)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:

            #Calculate area and remove small elements
            area = cv2.contourArea(cnt)
            if area > 100: 
                cv2.drawContours(RGBframe, [cnt], -1, (0,255,0), 2)

        cv2.imshow("frame", RGBframe)
        cv2.imshow("mask", mask)
        
        if cv2.waitKey(1) == ord('q'):
            break
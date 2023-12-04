import cv2 
import depthai 
import blobconverter
import numpy as np

pipeline = depthai.Pipeline()
cam = pipeline.create(depthai.node.ColorCamera)
cam.setColorOrder(depthai.ColorCameraProperties.ColorOrder.RGB)
cam.setPreviewSize(1280,720)
xout = pipeline.create(depthai.node.XLinkOut)
xout.setStreamName("rgb")
cam.preview.link(xout.input)


# Connect to device and start pipeline
with depthai.Device(pipeline) as device:
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    while True:
        #this frame is displayed in BGR
        BGRframe = qRgb.get().getFrame()
        #set frame to RGB
        RGBframe = cv2.cvtColor(BGRframe, cv2.COLOR_RGB2BGR)
        #display both frames
        cv2.imshow("RGB-Video", RGBframe)
        cv2.imshow("BGR-Video", BGRframe)
        if cv2.waitKey(1) == ord('q'):
            break
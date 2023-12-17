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

# Define your area threshold in square centimeters
area_threshold_cm2 = 5  # adjust this value based on your requirements

# Convert square centimeters to square pixels
pixels_per_micrometer = 1 / 1.55  # assuming 1.55µm x 1.55µm pixel size
area_threshold_pixels = (area_threshold_cm2 * 1e4) * pixels_per_micrometer**2

# Create a background subtractor with parameters
object_detector = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold = 40, detectShadows=False)

# Create a tracker
tracker = cv2.TrackerMIL.create()

# Connect to device and start pipeline
with depthai.Device(pipeline) as device:
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False) # type: ignore
    while True:
        #this frame is displayed in BGR
        BGRframe = qRgb.get().getFrame()
        #set frame to RGB
        RGBframe = cv2.cvtColor(BGRframe, cv2.COLOR_RGB2BGR)
        #set frame to RGB
        GRAYframe = cv2.cvtColor(BGRframe, cv2.COLOR_BGR2GRAY)

        #Object detection
        mask = object_detector.apply(RGBframe)
        contours, hierarchy  = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            #Calculate area and remove small elements
            area = cv2.contourArea(cnt)
            if area > area_threshold_pixels: 
                # cv2.drawContours(RGBframe, [cnt], -1, (0,255,0), 2)
                # Draw bounding box around the detected object
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(RGBframe, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Add label text at the top
                label_font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(RGBframe, 'Objekt', (x, y - 5), label_font, 0.9, (0, 255, 0), 2)

                # Add label text at the bottom
                cv2.putText(RGBframe, 'Objekt', (x, y + h + 25), label_font, 0.9, (0, 255, 0), 2)

                
        cv2.imshow("frame", RGBframe)
        cv2.imshow("mask", mask)
        
        if cv2.waitKey(1) == ord('q'):
            break


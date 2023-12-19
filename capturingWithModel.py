from pathlib import Path
import cv2 
import depthai 
import sys
import numpy as np
import cvlib as cv
import time
from cvlib.object_detection import draw_bbox

# Get argument first
nnPath = str((Path(__file__).parent / Path('yolo-v4-tiny-tf_openvino_2021.4_6shave.blob')).resolve().absolute())

# tiny yolo v4 label texts
labelMap = [
    "person",         "bicycle",    "car",           "motorbike",     "aeroplane",   "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",   "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",       "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",    "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball", "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",      "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",      "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",       "donut",         "cake",
    "chair",          "sofa",       "pottedplant",   "bed",           "diningtable", "toilet",        "tvmonitor",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",  "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",       "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush"
]

syncNN = True

# Create pipeline
pipeline = depthai.Pipeline()


# Define sources and outputs
cam = pipeline.create(depthai.node.ColorCamera)
detectionNetwork = pipeline.create(depthai.node.YoloDetectionNetwork)

xout = pipeline.create(depthai.node.XLinkOut)
nnOut = pipeline.create(depthai.node.XLinkOut)

xout.setStreamName("rgb")
nnOut.setStreamName("nn")

# Settings
cam.setColorOrder(depthai.ColorCameraProperties.ColorOrder.BGR)
cam.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setPreviewSize(416, 416)
cam.setInterleaved(False)
cam.setFps(40)
# Network specific settings
detectionNetwork.setConfidenceThreshold(0.5) # type: ignore
detectionNetwork.setNumClasses(80) # type: ignore
detectionNetwork.setCoordinateSize(4) # type: ignore
detectionNetwork.setAnchors([10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319]) # type: ignore
detectionNetwork.setAnchorMasks({"side26": [1, 2, 3], "side13": [3, 4, 5]}) # type: ignore
detectionNetwork.setIouThreshold(0.5) # type: ignore
detectionNetwork.setBlobPath(nnPath) # type: ignore
detectionNetwork.setNumInferenceThreads(2)
detectionNetwork.input.setBlocking(False)

# Linking
cam.preview.link(detectionNetwork.input)
if syncNN:
    detectionNetwork.passthrough.link(xout.input)
else:
    cam.preview.link(xout.input)
detectionNetwork.out.link(nnOut.input)


# Connect to device and start pipeline
with depthai.Device(pipeline) as device:

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False) # type: ignore
    qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False) # type: ignore

    frame = None
    detections = []
    startTime = time.monotonic()
    counter = 0
    color2 = (255, 255, 255)

    # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def displayFrame(name, frame):
        color = (255, 0, 0)
        for detection in detections:
            if (detection.label > 39 and detection.label < 55):
                bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                cv2.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(frame, f"{int(detection.confidence * 10)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                print(labelMap[detection.label])
                #save the positions of cutlery and donut
        # Show the frame 
        cv2.imshow(name, frame)

    while True:
        if syncNN:
            inRgb = qRgb.get()
            inDet = qDet.get()
        else:
            inRgb = qRgb.tryGet()
            inDet = qDet.tryGet()

        if inRgb is not None:
            frame = inRgb.getCvFrame()
            cv2.putText(frame, "NN fps: {:.2f}".format(counter / (time.monotonic() - startTime)),
                        (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color2)

        if inDet is not None:
            detections = inDet.detections
            counter += 1

        if frame is not None:
            displayFrame("rgb", frame)

        if cv2.waitKey(1) == ord('q'):
            break
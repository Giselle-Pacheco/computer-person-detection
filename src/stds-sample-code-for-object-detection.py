""" stds-sample-code-for-object-detection.py


    USAGE: 
    $ python stds-sample-code-for-object-detection.py --video_file football-field-cropped-video.mp4 --frame_resize_percentage 30

"""

# -----------------------  Import standard libraries ----------------------- #
from __future__ import print_function
import cv2 
import argparse
import numpy as np


# ------------------------- Define utility functions ----------------------- #
def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H-1, low_H)
    cv2.setTrackbarPos(low_H_name, window_detection_name, low_H)

def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H+1)
    cv2.setTrackbarPos(high_H_name, window_detection_name, high_H)

def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S-1, low_S)
    cv2.setTrackbarPos(low_S_name, window_detection_name, low_S)

def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S+1)
    cv2.setTrackbarPos(high_S_name, window_detection_name, high_S)

def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V-1, low_V)
    cv2.setTrackbarPos(low_V_name, window_detection_name, low_V)

def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V+1)
    cv2.setTrackbarPos(high_V_name, window_detection_name, high_V)


# ------------------ Define and initialise variables ---------------------- #
max_value = 255
max_value_H = 360//2
low_H = 75
low_S = 15
low_V = 0
high_H = 180
high_S = 170
high_V = 160 #max_value
window_capture_name = 'Input video'
window_detection_name = 'Object Detection'
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'


# ------------------ Parse data from the command line terminal ------------- #
parser = argparse.ArgumentParser(description='Vision-based object detection')
parser.add_argument('--video_file', type=str, default='camera', help='Video file used for the object detection process')
parser.add_argument('--frame_resize_percentage', type=float )
args = parser.parse_args()

# ------------------ Read video sequence file ------------------------------ #
cap = cv2.VideoCapture(args.video_file)
# filter=gamma_filter(cap,1.5)

# ------------- Create two new windows for visualisation purposes ---------- #
cv2.namedWindow(window_capture_name)
cv2.namedWindow(window_detection_name)

# ------------ Configure trackbars for the low and hight HSV values -------- #
cv2.createTrackbar(low_H_name, window_detection_name , low_H, max_value_H, on_low_H_thresh_trackbar)
cv2.createTrackbar(high_H_name, window_detection_name , high_H, max_value_H, on_high_H_thresh_trackbar)
cv2.createTrackbar(low_S_name, window_detection_name , low_S, max_value, on_low_S_thresh_trackbar)
cv2.createTrackbar(high_S_name, window_detection_name , high_S, max_value, on_high_S_thresh_trackbar)
cv2.createTrackbar(low_V_name, window_detection_name , low_V, max_value, on_low_V_thresh_trackbar)
cv2.createTrackbar(high_V_name, window_detection_name , high_V, max_value, on_high_V_thresh_trackbar)

# ------- Main loop --------------- #
while True:

    # Read current frame
    ret, frame = cap.read()

    # Check if the image was correctly captured
    if frame is None:
        break

    # Resize current frame
    width = int(frame.shape[1] * args.frame_resize_percentage / 100)
    height = int(frame.shape[0] * args.frame_resize_percentage / 100)
    dim = (width, height)
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    # Apply the median filter
    frame=cv2.medianBlur(frame,5)

    # frame = cv2.GaussianBlur(frame,(7,7),1)

    # Convert the current frame from BGR to HSV
    frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Apply a threshold to the HSV image
    frame_threshold = cv2.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))

    # Convert the current frame from BGR to Gray
    framem = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    # Filter out the grassy region from current frame and keep the moving object only
    # bitwise_AND = cv2.bitwise_and(frame_HSV, frame_HSV, mask=frame_threshold)
    bitwise_AND = cv2.bitwise_and(frame, frame, mask=frame_threshold)
    contours, hierarchy = cv2.findContours(frame_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area=cv2.contourArea(contour)
        max_value=np.max(bitwise_AND[:,:,1])
        # print('the max value is ', max_value)
        if max_value>110 and max_value<215:
            if area>80 and area<330:   
                perimeter=cv2.arcLength(contour,True)
                polynomio=cv2.approxPolyDP(contour,0.02*perimeter,True)
                # print(len(polynomio))
                x_,y_,w,h=cv2.boundingRect(polynomio)
                cv2.rectangle(bitwise_AND,(x_,y_),(x_+30,y_+30),(0,255,0),1)

    # Visualise both the input video and the object detection windows
    cv2.imshow(window_capture_name, frame)
    cv2.imshow(window_detection_name, bitwise_AND)

    # The program finishes if the key 'q' is pressed
    key = cv2.waitKey(5)
    if key == ord('q') or key == 27:
        print("Programm finished, mate!")
        break

# Destroy all visualisation windows
cv2.destroyAllWindows()

# Destroy 'VideoCapture' object
cap.release()
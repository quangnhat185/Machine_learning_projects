# -*- coding: utf-8 -*-
"""
Created on Sat May  9 12:11:20 2020

@author: Quang
"""

import os
import time

import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array

CONFIDENCE_THRESHOLD = 0.5

# Load face detector model
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join(["face_detector","deploy.prototxt"])
weightsPath = os.path.sep.join(["face_detector",
                                "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath,weightsPath)

# Load facemask detector from model
print("[INFO] loading facemask detector model...")
json_file = open('facemask_mobilenet.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
maskNet = model_from_json(loaded_model_json)
maskNet.load_weights("facemask_detection.h5") 

# create a function that detect face and deliver facemask prediction
def detect_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h,w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame,1.0,(300,300),(104.0,177.0,123.0))
    
    # pass the blob through the network and obtain the face detection
    faceNet.setInput(blob)
    detections = faceNet.forward()
    
    # initialize our list of faces, their corresponding locations,
    # and the list of predicitons from our face mask network
    faces = []
    locs = []
    
    # loop over the detection
    for i in range(0,detections.shape[2]):
        # extract the confidence associated with the detection
        confidence = detections[0,0,i,2]
        
        # filter out week detection with threshold confidence
        if confidence > CONFIDENCE_THRESHOLD:
            # compute the (x,y)-coordinates of the bounding box
            # for object
            box = detections[0,0,i,3:7]*np.array([w,h,w,h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # ensure the bounding boxes fall within the dimension 
            # of the frame
            (startX, startY) = (max(0,startX), max(0,startY))
            (endX, endY) = (min(w-1,endX),min(h-1,endY))
            
            # extract the face ROI, convert it to RGB
            # ordering, resize to 224x244 as model input size
            
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)
            
            # add the face and bounding boxes to their 
            # respective lists
            faces.append(face)
            locs.append((startX,startY,endX,endY))
            
    # only make prediction if at least one face was detected
    if len(faces)>0:
        pred = np.argmax(maskNet.predict(faces))
        
    return (locs, pred)

# initialize video stream
# cap = cv2.VideoCapture(1)   

# run on video    
cap = cv2.VideoCapture("facemask_video.mp4")
_, frame = cap.read()

# ratio = width / height
ratio = frame.shape[1]/frame.shape[0]
time.sleep(1.0)

# configure video recording
#writer = cv2.VideoWriter("videoname.mp4",cv2.VideoWriter_fourcc(*'DIVX'),24,(1024, int(1024/ratio)))
while True:
    
    _, frame = cap.read()
    
    if frame is None:
        break
    
    frame = cv2.resize(frame, (400, int(400/ratio)))
    
    # detect faces and predict facemask present
    try:
        (locs, pred) = detect_mask(frame, faceNet, maskNet)
        
        # loop over the detected face locations
        
        for box in locs:
            (startX, startY, endX, endY) = box  
            # determine class label 
            if pred==0:
                label = "Mask"
                color = (0, 255, 0)
            else:
                label = "No Mask"
                color = (0, 0, 255)
            
            # display the label and bounding box rectange on the output
            cv2.putText(frame, label, (startX, startY - 10),
                  cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            
    except Exception:
        pass
      
    # show the output frame
    frame_show = cv2.resize(frame,(1024, int(1024/ratio)))
#    writer.write(frame_show)
    cv2.imshow("Facemask prediction", frame_show)
    
    key = cv2.waitKey(10) & 0xFF
    
    if key == ord("q"):
        break

cv2.destroyAllWindows()            
cap.release()

# stop record video
# writer.release()

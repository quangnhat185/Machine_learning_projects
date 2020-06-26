# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 18:09:56 2020

@author: Quang
"""

"""
python age_classifier.py -v test_video.mp4 -s y
"""


import numpy as np
import argparse
import cv2
import os
from keras.models import model_from_json
from efficientnet import keras
import pickle
from bounding_box import bounding_box as bb

# setup argumentparse
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True, 
                help = "Path to video")
ap.add_argument("-f","--face", default="face_detector", 
                help  ="Path to face detector directory")
ap.add_argument("-m", "--model", default="model", 
                help = "Path to model directory")
ap.add_argument("-s","--save", default="n",
                help = "Save processed video (y/n)")
ap.add_argument("-c","--conf",default=0.3,
                help="Threshold fitering probability")
args = vars(ap.parse_args())



def resize(frame, new_width):
    (h,w) = frame.shape[:2]
    ratio = w/h
    new_frame = cv2.resize(frame, (new_width, int(new_width/ratio)))
    return new_frame
    

# retrun key from predicted value
def get_key(val, labels):
  for key, value in labels.items():
    if value==val: 
        if val==7:
            return "60_up"
        else:
            return key
        break
    
def get_age_from_model(face, model, labels):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face,(88,88))
    face = face/255
    face = face[np.newaxis,:] 
    val = np.argmax(model.predict(face))
    return get_key(val,labels)
  
def predict(frame,faceNet, model, labels):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > args["conf"]:
            box = detections[0, 0 , i, 3:7] * np.array([w, h , w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            face = frame[startY:endY, startX:endX]
            age  = get_age_from_model(face, model, labels)
            
    		# display the predicted age to our terminal
            text = "{:s}".format(age)
            
    		# draw the bounding box of the face along with the associated
    		# predicted age
            #y = startY - 10 if startY - 10 > 10 else startY + 10
            bb.add(frame,startX,startY,endX,endY,text)
            #cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
#            cv2.puttext(frame, text, (startX, y),cv2.font_hershey_simplex, 
#                        0.45, (0, 0, 255), 2)
            
    return frame        

def main():
    global record_status
    record_status = False
    
    WIDTH = 1024
    # load labels from trained model (age group)
    with open(os.path.join(args["model"],"labels.pkl"),"rb") as files:
        labels = pickle.load(files)

    # load face detector model 
    configPath = os.path.join(args["face"],"deploy.prototxt")
    weightsPath = os.path.join(args["face"],"res10_300x300_ssd_iter_140000.caffemodel")
    faceNet = cv2.dnn.readNet(configPath, weightsPath)
    print("[INFO] Load face detector model successfuly... ")
    
    # load age predictor model
    with open(os.path.join(args["model"],"age_prediction_efficientnet.json"),"r") as json_file:
        json_model = json_file.read()
    
    model = model_from_json(json_model)
    model.load_weights(os.path.join(args["model"],"age_prediction_efficientnet.h5"))
    print("[INFO] Load age prediciting model successfuly... ")
    
    cap = cv2.VideoCapture(args["video"])
    
    # save video according argument parser
    if args["save"] in ["y", "Y"]:
        record_status = True
        _, frame = cap.read()
        frame = resize(frame,WIDTH)
        (h,w) = frame.shape[:2]
        video_size = (w,h)
        writer = cv2.VideoWriter("processed_video.mp4",cv2.VideoWriter_fourcc(*'DIVX'),18,video_size)
  
    while True:
        try:
            _, frame = cap.read()
            
            if frame is None: 
                break
            
            # passing each frame through faceNet and age predicting model
            # then return processes frame
            frame = resize(frame,WIDTH)
            result = predict(frame, faceNet, model, labels)
                
            cv2.imshow("result", result)
            if record_status: writer.write(result) 
            key = cv2.waitKey(1) & 0xff
            
            # press ESC to exit
            if key == 27:
                break       
        except:
            pass
        
        
    if record_status: writer.release()
    cap.release()    
    cv2.destroyAllWindows()
    
    
if __name__=="__main__":
    main()    
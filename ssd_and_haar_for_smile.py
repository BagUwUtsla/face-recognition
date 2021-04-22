import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time 

video_capture = cv2.VideoCapture(0) 
detector = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel") 
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

seconds_last = int(time.time())
fps =  0

while(True):

    seconds_now = int( time.time() )

    ret, frame = video_capture.read()
    if not ret:
        break
    
    base_frame = frame.copy()
    
    # Haar cascade detection

    gray = cv2.cvtColor(base_frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(base_frame, (x, y), ((x + w), (y + h)), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = base_frame[y:y + h, x:x + w]
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
  
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2)


    # SSD detection

    original_size = frame.shape
    target_size = (300, 300)
    resized_frame = cv2.resize(frame, target_size)
    
    aspect_ratio_x = (original_size[1] / target_size[1])
    aspect_ratio_y = (original_size[0] / target_size[0])

    imageBlob = cv2.dnn.blobFromImage(image = resized_frame)
    detector.setInput(imageBlob)
    detections = detector.forward() 

    detections_df = pd.DataFrame(detections[0][0]
    , columns = ["img_id", "is_face", "confidence", "left", "top", "right", "bottom"])

    detections_df = detections_df[detections_df['is_face'] == 1] #0: background, 1: face
    detections_df = detections_df[detections_df['confidence'] >= 0.520]

    for i, instance in detections_df.iterrows():
        
        confidence_score = str(round(100*instance["confidence"], 2))+" %"
        
        left = int(instance["left"] * 300)
        bottom = int(instance["bottom"] * 300)
        right = int(instance["right"] * 300)
        top = int(instance["top"] * 300)
        
        detected_face = base_frame[int(top*aspect_ratio_y):int(bottom*aspect_ratio_y), int(left*aspect_ratio_x):int(right*aspect_ratio_x)]
        
        if detected_face.shape[0] > 0 and detected_face.shape[1] > 0:
            
            #write the confidence score above the face
            cv2.putText(base_frame, confidence_score, (int(left*aspect_ratio_x), int(top*aspect_ratio_y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            #draw rectangle to main image
            cv2.rectangle(base_frame, (int(left*aspect_ratio_x), int(top*aspect_ratio_y)), (int(right*aspect_ratio_x), int(bottom*aspect_ratio_y)), (255, 255, 255), 1) 
    
    for (x,y,w,h) in faces:
        base_frame = cv2.rectangle(base_frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = base_frame[y:y+h, x:x+w]
        smiles = smile_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in smiles:
            cv2.rectangle(base_frame,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('Video', base_frame)
    # if you press 'q' it will exit the programm
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    fps += 1 

    if seconds_last != seconds_now :
        print("Frames per second : {}".format(fps))
        fps = 0
        seconds_last = seconds_now

video_capture.release() 
cv2.destroyAllWindows() 
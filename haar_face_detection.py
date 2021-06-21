import cv2
import time

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
# To capture video from webcam. 
cap = cv2.VideoCapture(0)
# To use a video file as input 
# cap = cv2.VideoCapture('filename.mp4')
seconds_last = int(time.time())
fps = 0

while True:
    seconds_now = int( time.time() )

    # Read the frame
    _, img = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Detect smile
    smile = smile_cascade.detectMultiScale(gray, 1.7, 20)

    # Draw the rectangle around each face and smile
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    for (x, y, w, h) in smile: 
            cv2.rectangle(img,(x, y),(x+w, y+h), (255, 0, 130), 2)
    # Display
    cv2.imshow('img', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break

    if seconds_last != seconds_now :
        print("Frames per second : {}".format(fps))
        fps = 0
        seconds_last = seconds_now

# Release the VideoCapture object
cap.release()
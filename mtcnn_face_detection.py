from mtcnn import MTCNN
import cv2

video_capture = cv2.VideoCapture(0) 
detector = MTCNN()

while(True):

    ret, frame = video_capture.read()
    if not ret:
        break

    faces = detector.detect_faces(frame)

    for face in faces:
        if face["confidence"] > 0.8 :
            x, y, w, h = face["box"]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Video', frame)
    
    print(faces)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release() 
cv2.destroyAllWindows() 
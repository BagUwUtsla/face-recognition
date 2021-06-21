# import the necessary packages
from imutils import face_utils
import imutils
import dlib
import cv2
import math

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# launching the camera
video_capture = cv2.VideoCapture(0)

# creating a window
cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# is he smiling ? function


def smile_or_not(shape, frame):

    # getting the important points
    left_corner_mouth = shape[48]
    middle_mouth = shape[57]
    right_corner_mouth = shape[54]

    # power
    left_middle_x_pow = math.pow(middle_mouth[0] - left_corner_mouth[0], 2)
    left_middle_y_pow = math.pow(middle_mouth[1] - left_corner_mouth[1], 2)

    right_middle_x_pow = math.pow(
        middle_mouth[0] - right_corner_mouth[0], 2)
    right_middle_y_pow = math.pow(
        middle_mouth[1] - right_corner_mouth[1], 2)

    # square
    distance_left_middle = math.sqrt(left_middle_x_pow + left_middle_y_pow)
    distance_right_middle = math.sqrt(
        right_middle_x_pow + right_middle_y_pow)

    if (distance_right_middle > 35 and distance_left_middle > 35):
        text = "☺"
    else:
        text = "☻"

    cv2.putText(frame, text, (shape[8][0] - 40, shape[8][1]+10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame


# loop

while True:
    # grab the frame from the threaded video stream, resize it to
    # have a maximum width of 400 pixels, and convert it to
    # grayscale
    frame = video_capture.read()[1]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    # loop over the face detections
    for rect in rects:

        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        frame = smile_or_not(shape, frame)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        # for (x, y) in shape:
        #    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    # show the frame
    cv2.imshow("window", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
video_capture.release()
cv2.destroyAllWindows()

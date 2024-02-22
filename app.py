
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from keras.models import load_model
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

        
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)


def preprocess_and_reshape_mouth_region(frame, shape, margin=12, target_size=(64, 64)):
    x, y, w, h = cv2.boundingRect(np.array([shape[48:60]]))
    cv2.rectangle(frame, (x - margin, y - margin), (x + w + margin, y + h + margin), (0, 255, 0), 2)
    mouth_roi = frame[y - margin:y + h + margin, x - margin:x + w + margin]
    resized_mouth_roi = cv2.resize(mouth_roi, target_size)
    gray_mouth_roi = cv2.cvtColor(resized_mouth_roi, cv2.COLOR_BGR2GRAY)
    gray_mouth_roi = cv2.GaussianBlur(gray_mouth_roi, (3, 3), 0)
    normalized_mouth_roi = gray_mouth_roi.astype(np.float32) / 255.0
    input_image = np.reshape(normalized_mouth_roi, (1, target_size[0], target_size[1], 1))

    return input_image

def detect_yawn(frame, yawn_model, input_image, threshold=0.5):
    prediction = yawn_model.predict(input_image)
    if (prediction > threshold).astype(int):
        cv2.putText(frame, "YAWN ALERT!", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "NO YAWN", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)



ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0,
                help="index of webcam on system")
args = vars(ap.parse_args())

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 40
alarm_status = False
saying = False
COUNTER = 0

detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")    #Faster but less accurate
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
yawn_model = load_model("Yawnmodel_cnn (1).h5")

print("-> Starting Video Stream")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(0.5)


while True:

    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #rects = detector(gray, 0)
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
  minNeighbors=5, minSize=(30, 30),
  flags=cv2.CASCADE_SCALE_IMAGE)

    #for rect in rects:
    for (x, y, w, h) in rects:
        rect = dlib.rectangle(int(x), int(y), int(x + w),int(y + h))    
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        eye = final_ear(shape)
        ear = eye[0]
        leftEye = eye [1]
        rightEye = eye[2]
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # IDENTIFICAR MOUTH REGION
        #input_image = preprocess_and_reshape_mouth_region(frame, shape)
        
        # Make prediction
        #detect_yawn(frame, yawn_model,input_image)
        

        # -----------------------EYES------------
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER = 0
        



    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
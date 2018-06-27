# Captures the facial expression from webcam and predicts the emotion
import argparse
import sys, os
sys.path.append("../")

import cv2
import numpy as np

import FaceRecognitionUtility as objFaceRec

import model.CNN_2Conv as objModel

Screen_name = 'Real-time facial expression recognition'

FACE_SHAPE = (32, 32)

# While training model we have saved wights to this .h5 file. Now we will use those precalculated weights to build our model
model = objModel.buildModel('CNN(2Conv)_model_weights.h5')

final_emotions     = ["neutral", "anger", "fear", "happy", "sadness", "surprise"]

color = [(0,0,255),(0,0,255),(255,0,0),(0,127,255),(130,0,75),(0,255,0)]

def fnRefreshFrame(frame, faceCoordinates, index):
    if faceCoordinates is not None:
        if index is not None:
            cv2.putText (frame, final_emotions[index], (faceCoordinates[0] - 10, faceCoordinates[1] - 15),
                         cv2.FONT_HERSHEY_TRIPLEX, 2, color[index], 3, cv2.LINE_AA)
            objFaceRec.fnDrawFace (frame, faceCoordinates, color[index])
        else:
            objFaceRec.fnDrawFace (frame, faceCoordinates)

    cv2.imshow(Screen_name, frame)

# detects and predicts emotions from real time captured data
def fnShowAndDetect(capture):
    while (cv2.getWindowProperty('Real-time facial expression recognition', 0) >= 0):
        flag, frame = capture.read()
        faceCoordinates = objFaceRec.fnGetFaceCoordinates(frame)
        fnRefreshFrame(frame, faceCoordinates, None)
        
        if faceCoordinates is not None:
            face_img = objFaceRec.fnPreprocessImage(frame, faceCoordinates, face_shape=FACE_SHAPE)
            cv2.imshow(Screen_name, frame)
            objFaceRec.fnDrawFace(face_img, faceCoordinates)         
            input_img = np.array(face_img)
            input_img = input_img.reshape(-1, 32, 32, 1)

            result = model.predict(input_img)[0]
            index = np.argmax(result)
            print (final_emotions[index], 'prob:', max(result))
            fnRefreshFrame (frame, faceCoordinates, index)

        if cv2.waitKey(10) & 0xFF == 27:
            break

def fnGetCameraStreaming():
    capture = cv2.VideoCapture(0)
    if not capture:
        print("Failed to capture video streaming ")
        sys.exit(1)
    else:
        print("Successed to capture video streaming")
        
    return capture

#Arguments to be set:
# 1. showCam : determine if show the camera preview screen.
# if showCam  = 1 then it will show camera and if it 0, then it won't
def main():

    print("Enter main() function")  
    showCam = 1
    capture = fnGetCameraStreaming()

    if showCam:
        cv2.startWindowThread()
        cv2.namedWindow(Screen_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(Screen_name, cv2.WND_PROP_FULLSCREEN, cv2.WND_PROP_FULLSCREEN)
    
    fnShowAndDetect(capture)
    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

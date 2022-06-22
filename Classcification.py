from random import sample
import cv2
import mediapipe as mp 
import numpy as np 
import pandas as pd
import pickle
import math

counter = 0
stage = None


def Cal_Angle(landmark1 , landmark2 , landmark3):
    x1 , y1  = landmark1
    x2 , y2  = landmark2
    x3 , y3 = landmark3
    
    
    angle = math.degrees(math.atan2(y3-y2,x3-x2) - math.atan2(y1-y2,x1-x2) )
    if angle < 0 :
        angle += 360
        
        
    return angle


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
with open('Deadlift.pkl','rb') as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(0)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)
result = cv2.VideoWriter('filename.avi', cv2.VideoWriter_fourcc(*'MJPG'),10, size)



   









with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        try :
            landmarks = results.pose_landmarks.landmark
            row = np.array([[res.x,res.y,res.z,res.visibility] for res in landmarks]).flatten()
            X = pd.DataFrame([row])
            X = X.values
            deadlift_class = model.predict(X)[0]
            deadlift_prob =model.predict_proba(X)[0]
            # print(deadlift_class , deadlift_prob)
            Shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            Hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            Knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
           
            angle = Cal_Angle(Shoulder,Hip,Knee)
            if angle > 170 :
                stage = "up"
            if angle < 160 and stage =='up':
                stage="down"
                counter +=1


            cv2.rectangle(image,(0,0),(250,60),(255,0,255),-1)
            cv2.putText(image,'Class',(95,12),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.75,(0,0,0),1,cv2.LINE_AA)
            cv2.putText(image,deadlift_class.split(' ')[0],(95,40),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,0),1,cv2.LINE_AA)
            cv2.putText(image,'Prob',(15,12),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.75,(0,0,0),1,cv2.LINE_AA)
            cv2.putText(image,str(round(deadlift_prob[np.argmax(deadlift_prob)],2)),(12,40),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,0),2,cv2.LINE_AA)

           

            # Rep data
            cv2.putText(image, 'REPS', (170,12),cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter),(175,40),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,0), 1, cv2.LINE_AA)
        






        except:
            pass
        
        result.write(image)
        cv2.imshow('DeadLift_model', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()

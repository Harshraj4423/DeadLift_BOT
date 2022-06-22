import cv2
from cv2 import waitKey 
import mediapipe as mp 
import numpy as np 
import csv 


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def export_lnd(landmarks , class_name):
    
    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in landmarks]).flatten())
    pose_row.insert(0, class_name)

    with open('coords.csv', mode='a', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(pose_row) 

cap = cv2.VideoCapture('Video.mp4')
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
            pose_re = results.pose_landmarks.landmark
        except:
            pass
        k = cv2.waitKey(1)
        if k == 117:
            export_lnd(pose_re, 'up')


        if k == 104:
            export_lnd(pose_re, 'down')
        
        cv2.imshow('Image_show', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindow
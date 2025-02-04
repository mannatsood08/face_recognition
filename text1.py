import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import cv2
import csv
import time
from datetime import datetime
import os
from win32com.client import Dispatch

def speak(str1):
    speak = Dispatch(("SAPI.SpVoice"))
    speak.Speak(str1)

# Load labels and faces data
with open('data/names.pkl', 'rb') as f:
    LABELS = pickle.load(f)
with open('data/face_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

# Ensure data consistency
min_length = min(len(FACES), len(LABELS))
FACES = FACES[:min_length]
LABELS = LABELS[:min_length]

# Train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Load and resize the background image
imgBackground = cv2.imread("backgroundd.png")
COL_NAMES = ['NAMES', 'TIME']
resized_background_width = 1000  # New width
resized_background_height = 1200  # New height
imgBackground = cv2.resize(imgBackground, (resized_background_width, resized_background_height))

# Capture video from webcam
video = cv2.VideoCapture(0)
facesdetect = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facesdetect.detectMultiScale(gray, 1.3, 5)
    
    # Resize the frame to reduce its size
    reduced_frame = cv2.resize(frame, (400, 300)) 
    reduced_gray = cv2.cvtColor(reduced_frame, cv2.COLOR_BGR2GRAY)
    reduced_faces = facesdetect.detectMultiScale(reduced_gray, 1.3, 5)
    
    y_offset = 70  
    x_offset = 100  
    temp_frame = imgBackground.copy()

    for (x, y, w, h) in reduced_faces:
        crop_img = reduced_frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_img)
        ts = time.time()
        timestamp = datetime.fromtimestamp(ts).strftime("%H-%M-%S")
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        exist = os.path.isfile("Attendance/Attendance_" + date + ".csv")
        cv2.rectangle(reduced_frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.rectangle(reduced_frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
        cv2.rectangle(reduced_frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        cv2.putText(reduced_frame, str(output[0]), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        attendance = [str(output[0]), str(timestamp)]
        
        if exist:
            with open("Attendance/Attendance_" + date + ".csv", "a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(attendance)
        else:
            with open("Attendance/Attendance_" + date + ".csv", "a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                writer.writerow(attendance)

    if y_offset + reduced_frame.shape[0] <= temp_frame.shape[0] and x_offset + reduced_frame.shape[1] <= temp_frame.shape[1]:
        temp_frame[y_offset:y_offset+reduced_frame.shape[0], x_offset:x_offset+reduced_frame.shape[1]] = reduced_frame
    else:
        print("Reduced frame does not fit within the resized background at the specified coordinates.")

    cv2.imshow("Frame", temp_frame)
    k = cv2.waitKey(1)
 
    if k == ord('o'):
        speak("Attendance Taken")
        time.sleep(5)
        
    if k == ord('q'):
        break
video.release()
cv2.destroyAllWindows()

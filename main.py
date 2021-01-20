import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('./data/haarcascade_eye.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels = {}
with open("labels.pkl", "rb") as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()} #reverse labels

cap = cv2.VideoCapture(0)
while(True):
	#capture frame by frame
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5)
	for (x,y,w,h) in faces:
		cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
		roi_gray = gray[y:y+h, x:x+w]
		id_, conf = recognizer.predict(roi_gray)
		if conf >=50 and conf <=95:
			print(labels[id_])
			name = labels[id_]
			cv2.putText(frame, name, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,225), 2, cv2.LINE_AA)
	#display
	eyes = eye_cascade.detectMultiScale(gray)
	for (ex,ey,ew,eh) in eyes:
		cv2.rectangle(frame, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)
	cv2.imshow("frame1", frame)
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()
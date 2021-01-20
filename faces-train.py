import os
import numpy as np
from PIL import Image 
import cv2
import pickle
#to get path of this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#stores path to images folder
image_dir = os.path.join(BASE_DIR, "images")

face_cascade = cv2.CascadeClassifier("./data/haarcascade_frontalface_alt2.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0 #number associated with each label
label_ids = {}
y_labels = []
x_train = [] #captured trained data

for root, dirs, files in os.walk(image_dir):
	for file in files:
		#label is folders name
		if file.endswith("png") or file.endswith("jpg") or file.endswith("JPG"):
			path = os.path.join(root, file)
			label = os.path.basename(os.path.dirname(path))
			#print(label, path)
			if not label in label_ids:
				label_ids[label] = current_id
				current_id += 1
			id_ = label_ids[label]
			########ADD IMAGES TO NUMPY ARRAY#############
			pil_image = Image.open(path).convert("L") #grayscale conversion
			size = (550, 550) #sze to scale all images together
			final_image = pil_image.resize(size, Image.ANTIALIAS)
			image_array = np.array(final_image, "uint8")
			#print(image_array)
			########REGION OF INTEREST#############
			faces = face_cascade.detectMultiScale(image_array, 1.5, 5)

			for (x,y,w,h) in faces:
				roi = image_array[y:y+h, x:x+w]
				x_train.append(roi)
				y_labels.append(id_)
			########CREATING TRAINING LABELS#############
print(label_ids)
########USING PICKLE TO SAVE IDS#############
with open("labels.pkl", "wb") as f:
	pickle.dump(label_ids, f)
########TRAIN OPENCV RECOGNIZER#############
recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yml")
import cv2
import numpy as np
from PIL import Image
import os
path = 'dataset/'
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

image_paths = [os.path.join(path , f) for f in os.listdir(path)]

print(len(image_paths))
face_samples = []
index_samples = []
print(image_paths)

for image_path in image_paths:
    print(image_path)
    pil_image = Image.open(image_path).convert('L')
    np_image = np.array(pil_image , 'uint8')
    
    face = face_cascade.detectMultiScale(np_image)
    id = int(os.path.split(image_path)[-1].split(".")[0])
    
    for (x,y,w,h) in face:
        face_samples.append(np_image[y:y+h , x:x+w])
        index_samples.append(id)
        
len(face_samples)
len(index_samples)
recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.train(face_samples , np.array(index_samples))


recognizer.write('trainer/trainer.yml')

print("[INFO] {} faces trained . Exiting Program ".format(len(np.unique(index_samples))))


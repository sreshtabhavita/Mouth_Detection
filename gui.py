import tkinter as tk
from tkinter import filedialog
from tkinter import *
import os

from sklearn import metrics

from tensorflow.keras.models import model_from_json
from PIL import Image, ImageTk
import numpy as np
import cv2

# download haarcascade_frontalface_default from here "https://github.com/opencv/opencv/tree/master/data/haarcascades"

def MouthDetectionModel(json_file, weights_file):
    with open(json_file,"r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)

    model.load_weights(weights_file)
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

    return model

top = tk.Tk()
top.geometry('800x600')
top.title('Mouth open or close Detector')
top.configure(background='#CDCDCD')

label1 = Label(top, background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')

model = MouthDetectionModel("model_a1.json","model_weights1.h5")


MOUTH_LIST = ["Closed", "Open"]

def DetectMouth(file_path):
    global label_packed

    image = cv2.imread(file_path)
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    
    try:
        for (x,y,w,h) in faces:
            roi_gray = gray_image[y:y+h,x:x+w]
            roi_color = cv2.resize(roi_gray,(48,48))

            mouths = mouth_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25))

            for (mx, my, mw, mh) in mouths:
                mid_y = int(my + mh/2)
                cv2.rectangle(roi_color, (mx, my), (mx+mw, my+mh), (0, 255, 0), 2)

                roi = cv2.resize(roi_gray[my:my+mh, mx:mx+mw], (48, 48))
                pred = MOUTH_LIST[np.argmax(model.predict(roi_color[np.newaxis,:,:,np.newaxis]))]
        print("Prediction is " + pred)
        label1.configure(foreground="#011638", text = pred)
    except:
        label1.configure(foreground='#011638', text="Unable to detect")


def show_Detect_button(file_path):
    detect_b = Button(top,text="Detect Open/Close", command=lambda: DetectMouth(file_path),padx=10,pady=5)
    detect_b.configure(background="#364146", foreground='white', font=('arial',10,'bold'))
    detect_b.place(relx=0.79, rely=0.46)


def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image = im)
        sign_image.image = im
        label1.configure(text ='')
        show_Detect_button(file_path)
    except:
        pass


upload = Button(top,text = "Upload Image", command = upload_image, padx=10, pady=5)
upload.configure(background="#364156", foreground = 'white', font=('arial',20,'bold'))
upload.pack(side='bottom',pady=50)
sign_image.pack(side='bottom', expand='True')
label1.pack(side='bottom',expand='True')
heading=Label(top,text='MOUTH OPEN OR NOT DETECTION', pady=20, font=('arial',25,'bold'))
heading.configure(background='#CDCDCD', foreground="#364156")
heading.pack()
top.mainloop()

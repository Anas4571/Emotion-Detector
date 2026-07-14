import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam


emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))

# Load weights 
emotion_model.load_weights(r'D:\emoji-creator-project-code\emoji-creator-project-code\emotion_model.weights.h5')

cv2.ocl.setUseOpenCL(False)


emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
emoji_dist = {
    0: "./emojis/angry.png",
    1: "./emojis/disgusted.png",
    2: "./emojis/fearful.png",
    3: "./emojis/happy.png",
    4: "./emojis/neutral.png",
    5: "./emojis/sad.png",
    6: "./emojis/surprised.png"
}


last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
show_text = [0]


cap1 = cv2.VideoCapture(0)
if not cap1.isOpened():
    print("Cannot open camera")

def show_vid():
    ret, frame1 = cap1.read()
    if not ret:
        print("Failed to grab frame")
        lmain.after(10, show_vid)
        return

    frame1 = cv2.resize(frame1, (600, 500))
    bounding_box = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    num_faces = bounding_box.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame1, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray = gray_frame[y:y+h, x:x+w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48,48)), -1), 0)
        prediction = emotion_model.predict(cropped_img)
        show_text[0] = int(np.argmax(prediction))

    pic = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(pic)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_vid)  



def show_vid2():
    path = emoji_dist.get(show_text[0], None)
    if path is not None:
        frame2 = cv2.imread(path)
        if frame2 is not None:
            pic2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            img2 = Image.fromarray(pic2)
            imgtk2 = ImageTk.PhotoImage(image=img2)
            lmain2.imgtk2 = imgtk2
            lmain2.configure(image=imgtk2)
        else:
            print("Could not read emoji image:", path)
    else:
        print("Emoji path not found for index", show_text[0])

    lmain3.configure(text=emotion_dict.get(show_text[0], ""), font=('Arial', 45, 'bold'))
    lmain2.after(10, show_vid2)


if __name__ == '__main__':
    root = tk.Tk()
    heading = Label(root, text="Emotion Detector", bg='black', fg='white', font=('Arial', 45, 'bold'))
    heading.pack()

    lmain = tk.Label(master=root, padx=50, bd=10)
    lmain2 = tk.Label(master=root, bd=10)
    lmain3 = tk.Label(master=root, bd=10, fg="#CDCDCD", bg='black')

    lmain.pack(side=LEFT)
    lmain.place(x=50, y=250)
    lmain3.pack()
    lmain3.place(x=960, y=250)
    lmain2.pack(side=RIGHT)
    lmain2.place(x=900, y=350)

    root.title("Photo To Emoji")
    root.geometry("1400x900+100+10")
    root['bg'] = 'black'

    exitbutton = Button(root, text='Quit', fg="red", command=root.destroy, font=('Arial', 25, 'bold')).pack(side=BOTTOM)

    show_vid()
    show_vid2()
    root.mainloop()

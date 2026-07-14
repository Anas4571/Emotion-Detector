# # main.py
# import cv2
# import numpy as np
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
# from tensorflow.keras.optimizers import Adam

# # ---------------------------
# # Define the model architecture
# # ---------------------------
# emotion_model = Sequential()
# emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
# emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
# emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
# emotion_model.add(Dropout(0.25))

# emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
# emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
# emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
# emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
# emotion_model.add(Dropout(0.25))

# emotion_model.add(Flatten())
# emotion_model.add(Dense(1024, activation='relu'))
# emotion_model.add(Dropout(0.5))
# emotion_model.add(Dense(7, activation='softmax'))

# # Compile the model (necessary before loading weights)
# emotion_model.compile(
#     loss='categorical_crossentropy',
#     optimizer=Adam(learning_rate=0.0001, decay=1e-6),
#     metrics=['accuracy']
# )

# # Load trained weights
# emotion_model.load_weights('emotion_model.weights.h5')

# # Emotion labels
# emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 
#                 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# # ---------------------------
# # Start webcam feed
# # ---------------------------
# cap = cv2.VideoCapture(0)
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
#         roi_gray = gray_frame[y:y+h, x:x+w]
#         cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
#         prediction = emotion_model.predict(cropped_img)
#         maxindex = int(np.argmax(prediction))
#         cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

#     cv2.imshow('Emotion Detection', cv2.resize(frame, (1200, 860), interpolation=cv2.INTER_CUBIC))
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# # main.py - Pretrained Emotion Detection using FER
# import cv2
# from fer import FER

# # Initialize the webcam
# cap = cv2.VideoCapture(0)

# # Load pretrained emotion detector
# detector = FER(mtcnn=True)  # Use MTCNN for better face detection

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Detect emotions in the frame
#     result = detector.detect_emotions(frame)

#     for face in result:
#         (x, y, w, h) = face["box"]
#         emotions = face["emotions"]
#         # Get the emotion with the highest confidence
#         emotion = max(emotions, key=emotions.get)

#         # Draw rectangle around face
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
#         # Put emotion text above face
#         cv2.putText(frame, f"{emotion} ({emotions[emotion]:.2f})", (x, y-10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

#     # Display the webcam feed
#     cv2.imshow("Pretrained Emotion Detection", frame)

#     # Exit on pressing 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

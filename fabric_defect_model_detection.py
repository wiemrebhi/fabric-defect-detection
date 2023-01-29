import cv2
import numpy as np
from keras.models import load_model
from collections import deque

# Load the classifier model
model = load_model("textile.h5") #path to trained model
mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
Q = deque(maxlen=128)
lb=['free_stain','stain']
# Initialize video capture
cap = cv2.VideoCapture('video4-fabric2_edited.mp4') #path to video
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Check if the video has ended
    if not ret:
        break
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    frame = cv2.filter2D(frame, -1, kernel) # apply a sharpen filter

    # Preprocess the frame for the model
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224)).astype("float32")
    frame -= mean
    preds = model.predict(np.expand_dims(frame, axis=0))[0]
    print(preds)
    #Q.append(preds)
    #results = np.array(Q).mean(axis=0)
    i = np.argmax(preds)
    print(lb[i])
    if i == 1:
       #Apply background subtraction
       fgmask = fgbg.apply(frame)
    # Threshold the image to create a binary image
       ret, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
    # Find contours in the binary image
       contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       for c in contours:
            if cv2.contourArea(c) > 50:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Show the original frame with rectangles drawn around the defects
    cv2.imshow("Defects", frame)
    # Exit if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()

# Close all open windows
cv2.destroyAllWindows()

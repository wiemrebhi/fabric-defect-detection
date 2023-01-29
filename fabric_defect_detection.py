import cv2

# Load video
cap = cv2.VideoCapture('video4-fabric2_edited.mp4') #path to video

# Set up background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    # Read frame from video
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (700, 700))

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply background subtraction
    fgmask = fgbg.apply(gray)

    # Threshold the image to create a binary image
    ret, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw rectangles around contours that are larger than a certain size
    for c in contours:
        if cv2.contourArea(c) > 100:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Show the original frame with rectangles drawn around the defects
    cv2.imshow("Defects", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture object
cap.release()

# Close all windows
cv2.destroyAllWindows()

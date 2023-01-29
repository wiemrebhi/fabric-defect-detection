import cv2

# Load video
cap = cv2.VideoCapture("video4-fabric2_edited.mp4")#path to video

while True:
    # Read frame from video
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (700, 700))
    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds of the color of pink fabric in HSV color space
    lower_bound = (120, 50, 50)
    upper_bound = (150, 255, 255)

    # Define lower and upper bounds of the color of black stain in HSV color space
    black_lower = (0, 0, 0)
    black_upper = (180, 255, 30)

    # Create a binary mask for pink fabric
    pink_mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Create a binary mask for black stain
    black_mask = cv2.inRange(hsv, black_lower, black_upper)

    # Perform morphological operations to remove noise and fill in small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    pink_mask = cv2.morphologyEx(pink_mask, cv2.MORPH_CLOSE, kernel)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)

    # Use bitwise_and() function to combine the binary masks for pink fabric and black stains
    stains_mask = cv2.bitwise_and(pink_mask, black_mask)

    # Find contours in the mask
    contours, hierarchy = cv2.findContours(stains_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw rectangles around contours that are larger than a certain size
    for c in contours:
        if cv2.contourArea(c) > 0:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Show the original frame with rectangles drawn around the stains
    cv2.imshow("Stains", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture object
cap.release()

import cv2
import numpy as np
import pyautogui

# Set the resolution of the camera
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Set the speed of mouse movement
pyautogui.PAUSE = 0.01

# Set the minimum distance to move the mouse before considering it a drag
pyautogui.MINIMUM_DURATION = 0.2
pyautogui.MINIMUM_DISTANCE = 5

# Set the threshold values for detecting the hand
LOWER_THRESHOLD = np.array([0, 20, 70], dtype=np.uint8)
UPPER_THRESHOLD = np.array([20, 255, 255], dtype=np.uint8)

# Initialize the camera
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

while True:
    # Capture a frame from the camera
    ret, frame = camera.read()

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold the frame to detect the hand
    mask = cv2.inRange(hsv, LOWER_THRESHOLD, UPPER_THRESHOLD)

    # Find the contours of the hand in the mask
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If a hand is detected, move the mouse cursor
    if len(contours) > 0:
        # Find the largest contour, which is likely the hand
        largest_contour = max(contours, key=cv2.contourArea)

        # Find the centroid of the hand
        moments = cv2.moments(largest_contour)
        if moments["m00"] != 0:

            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])

        # Move the mouse cursor to the centroid of the hand
        pyautogui.moveTo(cx * 2, cy * 1.5)

        # If the area of the largest contour is large enough, simulate a left mouse button click
        if cv2.contourArea(largest_contour) > 2000:
            pyautogui.click(button='left')

    # Display the frame
    cv2.imshow('Virtual Mouse', frame)

    # If the 'q' key is pressed, exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
camera.release()
cv2.destroyAllWindows()

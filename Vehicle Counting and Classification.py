# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 22:54:35 2024

@author: amitKs
"""

import cv2
import numpy as np

# Load the video file
cap = cv2.VideoCapture('traffic.mp4')

# Initialize background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

# Variables for counting vehicles
counter_left = 0
counter_right = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame (e.g., resize, grayscale)
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(gray)

    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    # Find contours of vehicles
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Filter out small contours (noise)
        if cv2.contourArea(contour) < 500:
            continue

        # Get bounding box coordinates
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Determine the direction of the vehicle
        if x < frame.shape[1] // 2:
            counter_left += 1
        else:
            counter_right += 1

    # Display the frame and the counters
    cv2.putText(frame, "Left: " + str(counter_left), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "Right: " + str(counter_right), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Frame', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

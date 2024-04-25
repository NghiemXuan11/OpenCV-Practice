# import cv2
# import numpy as np

# # Load the input image
# input_image = cv2.imread('input_image3.jpg')

# # Initialize the SIFT detector
# sift = cv2.SIFT_create()

# # Detect keypoints and descriptors in the input image
# input_keypoints, input_descriptors = sift.detectAndCompute(
#     cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY), None)

# # Initialize video capture from webcam
# cap = cv2.VideoCapture(0)

# # Check if the webcam is opened successfully
# if not cap.isOpened():
#     print("Error: Unable to access webcam.")
#     exit()

# # FLANN parameters for matching
# FLANN_INDEX_KDTREE = 1
# index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
# search_params = dict(checks=50)

# # Create BFMatcher object
# bf = cv2.BFMatcher()

# # Initialize variables to store the first frame of the video
# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))
# video_frame = np.zeros((frame_height, frame_width, 3), np.uint8)

# # Loop to capture the first frame from the webcam
# while True:
#     # Read a frame from the webcam
#     ret, frame = cap.read()

#     # Check if the frame was successfully read
#     if ret:
#         video_frame = frame.copy()
#         break

# # Release the video capture object and close all windows
# cap.release()

# # Detect keypoints and descriptors in the first frame of the video
# video_keypoints, video_descriptors = sift.detectAndCompute(
#     cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY), None)

# # Match descriptors of input image with descriptors of the first frame of the video
# matches = bf.knnMatch(input_descriptors, video_descriptors, k=2)

# # Draw SIFT matching lines between keypoints of input image and the first frame of the video
# good_matches = []
# for m, n in matches:
#     if m.distance < 0.7 * n.distance:
#         good_matches.append(m)

# # Draw SIFT matching lines on the input image and the first frame of the video
# output_image = cv2.drawMatches(input_image, input_keypoints, video_frame, video_keypoints,
#                                good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# # Display the output image
# cv2.imshow('SIFT Matching between Image and Video Frame', output_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# ========================================================================================================


# import cv2
# import numpy as np

# # Load the input image
# input_image = cv2.imread('input_image3.jpg')

# # Initialize the SIFT detector
# sift = cv2.SIFT_create()

# # Detect keypoints and descriptors in the input image
# input_keypoints, input_descriptors = sift.detectAndCompute(
#     cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY), None)

# # Function to detect and draw rectangles around faces


# def detect_and_draw_faces(frame):
#     # Convert the frame to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Load pre-trained face detector
#     face_cascade = cv2.CascadeClassifier(
#         cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#     # Detect faces in the frame
#     faces = face_cascade.detectMultiScale(
#         gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     # Draw green rectangles around the detected faces
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

#     return frame


# # Initialize video capture from webcam
# cap = cv2.VideoCapture(0)

# # Check if the webcam is opened successfully
# if not cap.isOpened():
#     print("Error: Unable to access webcam.")
#     exit()

# # FLANN parameters for matching
# FLANN_INDEX_KDTREE = 1
# index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
# search_params = dict(checks=50)

# # Create BFMatcher object
# bf = cv2.BFMatcher()

# # Initialize variables to store the first frame of the video
# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))
# video_frame = np.zeros((frame_height, frame_width, 3), np.uint8)

# # Loop to capture the first frame from the webcam
# while True:
#     # Read a frame from the webcam
#     ret, frame = cap.read()

#     # Check if the frame was successfully read
#     if ret:
#         video_frame = frame.copy()
#         break

# # Release the video capture object
# cap.release()

# # Detect keypoints and descriptors in the first frame of the video
# video_keypoints, video_descriptors = sift.detectAndCompute(
#     cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY), None)

# # Match descriptors of input image with descriptors of the first frame of the video
# matches = bf.knnMatch(input_descriptors, video_descriptors, k=2)

# # Draw SIFT matching lines between keypoints of input image and the first frame of the video
# good_matches = []
# for m, n in matches:
#     if m.distance < 0.7 * n.distance:
#         good_matches.append(m)

# # Draw SIFT matching lines on the input image and the first frame of the video
# output_image = cv2.drawMatches(input_image, input_keypoints, video_frame, video_keypoints,
#                                good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# # Detect and draw rectangles around faces in the input image
# output_image_with_faces = detect_and_draw_faces(output_image)

# # Display the output image
# cv2.imshow('SIFT Matching with Face Detection', output_image_with_faces)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ===========================================================================================

import cv2
import numpy as np

# Initialize video capture from webcam
cap = cv2.VideoCapture(0)

# Load pre-trained face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Unable to access webcam.")
    exit()

# Capture the first frame from the webcam
ret, frame = cap.read()

# Check if the frame was successfully read
if not ret:
    print("Error: Unable to read frame.")
    exit()

# Convert the first frame to grayscale for face detection
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale frame
faces = face_cascade.detectMultiScale(
    gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Draw green rectangles around the detected faces in the first frame
for (x, y, w, h) in faces:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Load the input image
input_image = cv2.imread('input_image3.jpg')

# Initialize the SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and descriptors in the input image
input_keypoints, input_descriptors = sift.detectAndCompute(
    cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY), None)

# Detect keypoints and descriptors in the first frame
frame_keypoints, frame_descriptors = sift.detectAndCompute(gray_frame, None)

# Create BFMatcher object
bf = cv2.BFMatcher()

# Match descriptors of input image with descriptors of the first frame
matches = bf.knnMatch(input_descriptors, frame_descriptors, k=2)

# Draw SIFT matching lines between keypoints of input image and the first frame
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# Draw SIFT matching lines on the input image and the first frame
output_image = cv2.drawMatches(input_image, input_keypoints, frame, frame_keypoints,
                               good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Draw a green rectangle around the detected faces in the input image
# for (x, y, w, h) in faces:
#     cv2.rectangle(output_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Display the resulting image
cv2.imshow('SIFT Matching with Face Detection', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

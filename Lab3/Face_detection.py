# import numpy as np
# import cv2 as cv
# import matplotlib.pyplot as plt

# while cap.isOpened():
#     _,frame = cap.read()
#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     keypoints_2, descriptors_2 = sift.detectAndCompute(gray,None)
#     matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)
#     good = []
#     for m, n in matches:
#         if m.distance <= 0.9 * n.distance:
#             good.append([m])

#     img3 = cv.drawMatchesKnn(img, keypoints_1, gray, keypoints_2, good, None,
#                               matchColor = (0,255,0), matchesMask = None,
#                               singlePointColor = (255,0,0), flags=0)
#     cv2.imshow('Plann Match',img3)
#     if len(good) >= 10:
#         src_pts = np.float32([keypoints_1[m[0].queryIdx].pt for m in good]).reshape(-1, 1, 2)
#         dst_pts = np.float32([keypoints_2[m[0].trainIdx].pt for m in good]).reshape(-1, 1, 2)


import cv2

# Load the cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect and draw rectangles around faces using SIFT descriptors


def detect_and_draw_faces(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop through each face and draw a green rectangle around it
    for (x, y, w, h) in faces:
        # Draw a green rectangle around the face
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Return the image with rectangles drawn around the faces
    return image


# Load the input image
input_image = cv2.imread('box_in_scene2.jpg')

# Detect faces and draw rectangles around them
output_image = detect_and_draw_faces(input_image)

# Display the output image
cv2.imshow('Face Detection', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

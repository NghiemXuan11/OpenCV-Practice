import cv2
import numpy as np

# Load the Coca-Cola logo image
coca_cola_logo = cv2.imread('D:\\CV\\Lab2\\Coca.png', cv2.IMREAD_GRAYSCALE)  # Provide the path to the Coca-Cola logo image

# Initialize the SIFT detector
sift = cv2.SIFT_create()

# Find keypoints and descriptors in the Coca-Cola logo image
kp1, des1 = sift.detectAndCompute(coca_cola_logo, None)

# Create a FLANN matcher
flann = cv2.FlannBasedMatcher({'algorithm': 0, 'trees': 5}, {'checks': 50})

# Load the video file
cap = cv2.VideoCapture('D:\\CV\\Lab2\\Coca.mp4')  # Provide the path to your video file

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find keypoints and descriptors in the frame
    kp2, des2 = sift.detectAndCompute(gray_frame, None)

    if des2 is not None:
        # Match descriptors between the logo image and the frame
        matches = flann.knnMatch(des1, des2, k=2)

        # Filter good matches using Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        # If enough good matches are found, draw the bounding box around the logo
        if len(good_matches) > 8:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matches_mask = mask.ravel().tolist()

            h, w = coca_cola_logo.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3)

    # Display the frame with the detected Coca-Cola logo
    cv2.imshow('Coca-Cola Logo Detection', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

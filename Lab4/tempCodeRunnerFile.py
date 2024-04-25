import cv2
import numpy as np

def stitch_images(img1, img2, img3):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and descriptors
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    kp3, des3 = sift.detectAndCompute(gray3, None)

    # Match keypoints
    bf = cv2.BFMatcher()
    matches1_2 = bf.knnMatch(des1, des2, k=2)
    matches2_3 = bf.knnMatch(des2, des3, k=2)

    # Apply ratio test
    good_matches1_2 = []
    for m, n in matches1_2:
        if m.distance < 0.75 * n.distance:
            good_matches1_2.append(m)

    good_matches2_3 = []
    for m, n in matches2_3:
        if m.distance < 0.75 * n.distance:
            good_matches2_3.append(m)

    # Estimate homographies
    src_pts1_2 = np.float32(
        [kp1[m.queryIdx].pt for m in good_matches1_2]).reshape(-1, 1, 2)
    dst_pts1_2 = np.float32(
        [kp2[m.trainIdx].pt for m in good_matches1_2]).reshape(-1, 1, 2)
    H1_2, _ = cv2.findHomography(src_pts1_2, dst_pts1_2, cv2.RANSAC, 5.0)

    src_pts2_3 = np.float32(
        [kp2[m.queryIdx].pt for m in good_matches2_3]).reshape(-1, 1, 2)
    dst_pts2_3 = np.float32(
        [kp3[m.trainIdx].pt for m in good_matches2_3]).reshape(-1, 1, 2)
    H2_3, _ = cv2.findHomography(src_pts2_3, dst_pts2_3, cv2.RANSAC, 5.0)

    # Warp images
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h3, w3 = img3.shape[:2]
    warped_img2 = cv2.warpPerspective(img2, H1_2, (w1+w2, h1))
    warped_img3 = cv2.warpPerspective(img3, H2_3, (w2+w3, h2))

    # Resize warped images to match expected dimensions
    warped_img2 = cv2.resize(warped_img2, (w1, h1))
    warped_img3 = cv2.resize(warped_img3, (w3, h3))

    # Combine images
    panorama = np.zeros((max(h1, h3), w1+w2+w3, 3), dtype=np.uint8)
    panorama[:h1, :w1] = img1
    panorama[:h2, w1:w1+w2] = warped_img2
    panorama[:h3, w1+w2:] = warped_img3

    return panorama


# Load images
img1 = cv2.imread('images/3_images/first.jpg')
img2 = cv2.imread('images/3_images/second.jpg')
img3 = cv2.imread('images/3_images/third.jpg')

# Check if images are loaded successfully
if img1 is None or img2 is None or img3 is None:
    print("Error: One or more images could not be loaded.")
    exit()

# Stitch images
panorama = stitch_images(img1, img2, img3)

# Scale result
scale_percent = 30  # percent of original size
width = int(panorama.shape[1] * scale_percent / 100)
height = int(panorama.shape[0] * scale_percent / 100)
dim = (width, height)
resized_panorama = cv2.resize(panorama, dim, interpolation=cv2.INTER_AREA)

# Display panorama
cv2.imshow('Panorama', resized_panorama)
cv2.waitKey(0)
cv2.destroyAllWindows()
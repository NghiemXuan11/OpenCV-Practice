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
img1 = cv2.imread('img02.jpg')
img2 = cv2.imread('img03.jpg')
img3 = cv2.imread('img04.jpg')

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


#============================================================================================================

# import numpy as np
# from PIL import Image

# def align_images(image1, image2, image3):
#     # Implement image alignment (optional)
#     # For simplicity, let's assume the images are already aligned

#     return image1, image2, image3

# def stitch_images(image1, image2, image3):
#     # Align images (optional)
#     image1, image2, image3 = align_images(image1, image2, image3)

#     # Convert images to numpy arrays
#     image1_array = np.array(image1)
#     image2_array = np.array(image2)
#     image3_array = np.array(image3)

#     # Calculate dimensions of the stitched image
#     total_width = image1_array.shape[1] + image2_array.shape[1] + image3_array.shape[1]
#     max_height = max(image1_array.shape[0], image2_array.shape[0], image3_array.shape[0])

#     # Create a blank canvas for the stitched image
#     stitched_image = np.zeros((max_height, total_width, 3), dtype=np.uint8)

#     # Paste the first image onto the canvas
#     stitched_image[:image1_array.shape[0], :image1_array.shape[1], :] = image1_array

#     # Calculate the starting point for pasting the second image
#     start_x2 = image1_array.shape[1]
#     end_x2 = start_x2 + image2_array.shape[1]

#     # Paste the second image onto the canvas
#     stitched_image[:image2_array.shape[0], start_x2:end_x2, :] = image2_array

#     # Calculate the starting point for pasting the third image
#     start_x3 = end_x2
#     end_x3 = start_x3 + image3_array.shape[1]

#     # Paste the third image onto the canvas
#     stitched_image[:image3_array.shape[0], start_x3:end_x3, :] = image3_array

#     # Convert the numpy array back to an image
#     panorama = Image.fromarray(stitched_image)

#     return panorama

# # Load images
# image1 = Image.open('images\\3_images\\first.jpg')
# image2 = Image.open('images\\3_images\\second.jpg')
# image3 = Image.open('images\\3_images\\third.jpg')

# # Stitch the images together
# panorama = stitch_images(image1, image2, image3)

# # Save the panorama image
# panorama.save('panorama.jpg')


#===============================================================================================




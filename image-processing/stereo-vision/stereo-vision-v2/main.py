import numpy as np
import cv2
import matplotlib.pyplot as plt

imgLeft = cv2.imread('orange_left.jpg', 0)
imgRight = cv2.imread('orange_right.jpg', 0)

# Apply Gaussian blur to both images
imgLeft = cv2.GaussianBlur(imgLeft, (5, 5), 0)
imgRight = cv2.GaussianBlur(imgRight, (5, 5), 0)

# Downscale the images by a factor of 0.5 using INTER_AREA interpolation
imgLeft = cv2.resize(imgLeft, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
imgRight = cv2.resize(imgRight, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

imgLeft = cv2.GaussianBlur(imgLeft, (5, 5), 0)
imgRight = cv2.GaussianBlur(imgRight, (5, 5), 0)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(imgLeft, cmap='gray')
plt.title('Left Image')
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(imgRight, cmap='gray')
plt.title('Right Image')
plt.axis('off')
plt.show()

def ShowDisparity(bSize=5):
    # Initialize the stereo block matching object
    stereo = cv2.StereoBM_create(numDisparities=256 , blockSize=bSize)
    cv2.stereo
    # Compute the disparity image
    disparity = stereo.compute(imgLeft, imgRight)

    # Normalize the image for representation
    min = disparity.min()
    max = disparity.max()
    disparity= np.uint8(255 * (disparity - min) /(max -min))

    # Plot the result
    return disparity


result = ShowDisparity(15)
plt.imshow(result, cmap='gray')
plt.axis('off')
plt.show()
import cv2
import math
import numpy as np

def calculate_distance(img_width_L, img_width_R, obj_pos_L, obj_pos_R, fov_degrees, baseline):
    # Convert FOV from degrees to radians
    fov_rad = math.radians(fov_degrees)
    
    # Calculate angle per pixel for each image
    angle_per_pixel_L = fov_rad / img_width_L
    angle_per_pixel_R = fov_rad / img_width_R
    
    # Calculate horizontal displacement from center for each object
    x_center_L = img_width_L / 2
    x_center_R = img_width_R / 2
    
    # Left camera: displacement from center (positive if object is to the right)
    delta_x_L = obj_pos_L[0] - x_center_L
    
    # Right camera: displacement from center (positive if object is to the left)
    delta_x_R = x_center_R - obj_pos_R[0]  # Corrected sign
    
    # Calculate angles for each camera (in radians)
    theta_L = delta_x_L * angle_per_pixel_L
    theta_R = delta_x_R * angle_per_pixel_R
    
    # Calculate distance using triangulation formula
    distance = baseline / (math.tan(theta_L) + math.tan(theta_R))
    
    return distance

# Example usage:
# Load images
imgL = cv2.imread('orange_left.jpg')
imgR = cv2.imread('orange_right.jpg')

# Convert to grayscale
grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

# Detect circles using Hough Circle Transform
def detect_circle(image):
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                               param1=50, param2=30, minRadius=10, maxRadius=100)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        return (circles[0, 0][0], circles[0, 0][1])
    else:
        raise ValueError("No circles detected")

try:
    circleL = detect_circle(grayL)
    circleR = detect_circle(grayR)
    
    # Assuming both images have the same width and FOV
    img_width = grayL.shape[1]
    fov = 60  # Example FOV in degrees
    baseline = 0.1  # Example baseline in meters (adjust according to your setup)
    
    distance = calculate_distance(img_width, img_width, circleL, circleR, fov, baseline)
    print(f"Distance to object: {distance:.2f} meters")
except ValueError as e:
    print(e)
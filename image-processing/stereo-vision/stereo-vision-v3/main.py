import cv2
import numpy as np
import math
import os

def find_yellow_object(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    lower_yellow = np.array([12, 80, 10])
    upper_yellow = np.array([36, 255, 255])
    
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return [cx, cy, int(np.sqrt(cv2.contourArea(largest_contour)/np.pi))]
    return None

def calculate_distance(img_width, x_left, x_right, fov_degrees, baseline):
    disparity = abs(x_right - x_left)
    distance = baseline*img_width/(2*disparity*math.tan(math.radians(fov_degrees/2)))
    return distance, disparity

def process_image_pair(folder_path):
    img_left = cv2.imread(os.path.join(folder_path, 'left.jpg'))
    img_right = cv2.imread(os.path.join(folder_path, 'right.jpg'))
    
    if img_left is None or img_right is None:
        print(f"Error loading images from {folder_path}")
        return

    scale_percent = 75
    img_left = cv2.resize(
        img_left,
        (int(img_left.shape[1] * scale_percent / 100), int(img_left.shape[0] * scale_percent / 100)),
        interpolation=cv2.INTER_AREA
    )
    img_right = cv2.resize(
        img_right,
        (int(img_right.shape[1] * scale_percent / 100), int(img_right.shape[0] * scale_percent / 100)),
        interpolation=cv2.INTER_AREA
    )

    img_left = cv2.GaussianBlur(img_left, (5, 5), 0)
    img_right = cv2.GaussianBlur(img_right, (5, 5), 0)

    yellow_left = find_yellow_object(img_left)
    yellow_right = find_yellow_object(img_right)
    
    if yellow_left is None or yellow_right is None:
        print(f"No yellow objects found in one or both images in {folder_path}")
        return
    
    x_left = yellow_left[0]
    x_right = yellow_right[0]
    
    distance, disparity = calculate_distance(img_left.shape[1], x_left, x_right, 50, 0.2)
    
    if distance:
        offset = 20
        text_left = f"Dist: {distance:.3f}m"
        text_right = f"Dist: {distance:.3f}m"
        
        cv2.putText(
            img_left,
            text_left,
            (yellow_left[0] - 40, yellow_left[1] + yellow_left[2] + offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
        cv2.putText(
            img_right,
            text_right,
            (yellow_right[0] - 40, yellow_right[1] + yellow_right[2] + offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
        print(f"Results for {folder_path}:")
        print(f"  - Estimated distance: {distance:.3f} meters")
        print(f"  - Disparity: {disparity} pixels")
        print()
    
    cv2.circle(img_left, (yellow_left[0], yellow_left[1]), yellow_left[2], (0, 255, 0), 2)
    cv2.circle(img_right, (yellow_right[0], yellow_right[1]), yellow_right[2], (0, 255, 0), 2)
    
    window_name_left = f"Left Image - {os.path.basename(folder_path)}"
    window_name_right = f"Right Image - {os.path.basename(folder_path)}"
    
    cv2.imshow(window_name_left, img_left)
    cv2.imshow(window_name_right, img_right)



folders = ['10m', '20m']

for folder in folders:
    process_image_pair(folder)

cv2.waitKey(0)
cv2.destroyAllWindows()
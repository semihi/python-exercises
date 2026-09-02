import cv2
import numpy as np
import math

def find_circle(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=20,
        maxRadius=100
    )
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        
        return circles[0]
    return None

def calculate_distance(img_width, x_left, x_right, fov_degrees, baseline):
    
    distance = baseline*img_width/(2*(abs(x_right-x_left))*math.tan(math.radians(fov_degrees/2)))
    
    return distance

def main():
    
    img_left = cv2.imread('orange_left.jpg')
    img_right = cv2.imread('orange_right.jpg')
    
    if img_left is None or img_right is None:
        print("Error loading images")
        return


    scale_percent = 50
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


    circle_left = find_circle(img_left)
    circle_right = find_circle(img_right)
    
    if circle_left is None or circle_right is None:
        print("No circles found in one or both images")
        return
    
    
    x_left = circle_left[0]
    x_right = circle_right[0]
    
    
    distance = calculate_distance(img_left.shape[1], x_left, x_right, 65.644, 0.1)
    
    if distance:
        offset = 20
        cv2.putText(
            img_left,
            f"{distance:.3f} m",
            (circle_left[0] - 40, circle_left[1] + circle_left[2] + offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
        cv2.putText(
            img_right,
            f"{distance:.3f} m",
            (circle_right[0] - 40, circle_right[1] + circle_right[2] + offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
        print(f"Estimated distance to object: {distance:.3f} meters")
    
    
    cv2.circle(img_left, (circle_left[0], circle_left[1]), circle_left[2], (0, 255, 0), 2)
    cv2.circle(img_right, (circle_right[0], circle_right[1]), circle_right[2], (0, 255, 0), 2)
    
    
    cv2.imshow("Left Image", img_left)
    cv2.imshow("Right Image", img_right)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

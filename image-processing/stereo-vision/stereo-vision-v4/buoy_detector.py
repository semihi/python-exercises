import cv2
import numpy as np
from config import BuoyDetectorConfig

class BuoyDetector:
    """
    A class for detecting buoys in images with varying lighting conditions
    such as shadows, shine effects, and normal lighting.
    """
    
    def __init__(self, config=None):
        """Initialize the buoy detector with configuration parameters"""
        self.config = config if config else BuoyDetectorConfig()
        
    def preprocess_image(self, image):
        """Preprocess the image to handle varying lighting conditions"""
        # Convert to grayscale (if needed)
        if self.config.use_grayscale:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Apply histogram equalization
            if self.config.use_histogram_equalization:
                gray = cv2.equalizeHist(gray)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to reduce noise
        if self.config.use_gaussian_blur:
            gray = cv2.GaussianBlur(gray, 
                                    (self.config.gaussian_kernel_size, self.config.gaussian_kernel_size), 
                                    self.config.gaussian_sigma)
        
        # Apply adaptive thresholding if needed
        if self.config.use_adaptive_threshold and self.config.use_grayscale:
            binary = cv2.adaptiveThreshold(gray, 
                                          255, 
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 
                                          self.config.adaptive_block_size, 
                                          self.config.adaptive_c)
            return gray, binary
            
        return gray, None
    
    def color_thresholding(self, image):
        """Apply color thresholding in different color spaces"""
        masks = []
        
        # HSV color space thresholding
        if self.config.use_hsv_thresholding:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            for color_range in self.config.hsv_color_ranges:
                lower = np.array(color_range['lower'])
                upper = np.array(color_range['upper'])
                mask = cv2.inRange(hsv, lower, upper)
                masks.append(mask)
        
        # LAB color space thresholding
        if self.config.use_lab_thresholding:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            for color_range in self.config.lab_color_ranges:
                lower = np.array(color_range['lower'])
                upper = np.array(color_range['upper'])
                mask = cv2.inRange(lab, lower, upper)
                masks.append(mask)
        
        # Combine masks if multiple are created
        if masks:
            combined_mask = masks[0]
            for mask in masks[1:]:
                combined_mask = cv2.bitwise_or(combined_mask, mask)
            
            # Apply morphological operations to clean up the mask
            if self.config.use_morphological_operations:
                kernel = np.ones((self.config.morphological_kernel_size, 
                                 self.config.morphological_kernel_size), 
                                np.uint8)
                combined_mask = cv2.morphologyEx(combined_mask, 
                                               cv2.MORPH_CLOSE, 
                                               kernel, 
                                               iterations=self.config.close_iterations)
                combined_mask = cv2.morphologyEx(combined_mask, 
                                               cv2.MORPH_OPEN, 
                                               kernel, 
                                               iterations=self.config.open_iterations)
            return combined_mask
        
        return None
    
    def detect_contours(self, binary_image):
        """Detect and filter contours based on shape and size criteria"""
        contours, _ = cv2.findContours(binary_image, 
                                      cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        
        filtered_contours = []
        for contour in contours:
            # Filter by area
            area = cv2.contourArea(contour)
            if area < self.config.min_contour_area or area > self.config.max_contour_area:
                continue
            
            # Filter by shape (circularity/compactness)
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < self.config.min_circularity:
                continue
            
            # Filter by aspect ratio using bounding rect
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h != 0 else 0
            if (aspect_ratio < self.config.min_aspect_ratio or 
                aspect_ratio > self.config.max_aspect_ratio):
                continue
                
            filtered_contours.append(contour)
            
        return filtered_contours
    
    def detect_circles(self, gray_image):
        """Detect circles using Hough Circle Transform"""
        if not self.config.use_hough_circles:
            return []
            
        circles = cv2.HoughCircles(gray_image, 
                                  cv2.HOUGH_GRADIENT, 
                                  dp=self.config.hough_dp, 
                                  minDist=self.config.hough_min_dist,
                                  param1=self.config.hough_param1, 
                                  param2=self.config.hough_param2,
                                  minRadius=self.config.hough_min_radius, 
                                  maxRadius=self.config.hough_max_radius)
        
        if circles is not None:
            return circles[0]
        return []
    
    def calculate_buoy_properties(self, contours):
        """Calculate properties (centroid, width) for each detected buoy"""
        buoys = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Calculate width from bounding box or fitted ellipse
                if self.config.use_ellipse_fitting and len(contour) >= 5:
                    ellipse = cv2.fitEllipse(contour)
                    width = min(ellipse[1])  # Minor axis length
                else:
                    x, y, w, h = cv2.boundingRect(contour)
                    width = min(w, h)
                
                buoys.append({
                    'center': (cx, cy),
                    'width': width,
                    'contour': contour
                })
                
        return buoys
    
    def detect_buoys(self, image):
        """Main function to detect buoys in an image"""
        # Make a copy of the original image
        original = image.copy()
        
        # Preprocess the image
        preprocessed, binary = self.preprocess_image(image)
        
        # Apply color thresholding
        color_mask = self.color_thresholding(image)
        
        # Combine binary images if both are available
        if binary is not None and color_mask is not None:
            combined_binary = cv2.bitwise_or(binary, color_mask)
        elif binary is not None:
            combined_binary = binary
        elif color_mask is not None:
            combined_binary = color_mask
        else:
            # If no binary masks, use edge detection
            edges = cv2.Canny(preprocessed, 
                            self.config.canny_threshold1, 
                            self.config.canny_threshold2)
            # Dilate edges to close contours
            kernel = np.ones((3, 3), np.uint8)
            combined_binary = cv2.dilate(edges, kernel, iterations=1)
        
        # Detect contours
        contours = self.detect_contours(combined_binary)
        
        # Detect circles if enabled
        circles = []
        if self.config.use_grayscale and self.config.use_hough_circles:
            circles = self.detect_circles(preprocessed)
        
        # Calculate buoy properties from contours
        buoys = self.calculate_buoy_properties(contours)
        
        # Add circles to buoys if they don't overlap with existing contours
        for circle in circles:
            x, y, r = circle
            is_new = True
            for buoy in buoys:
                cx, cy = buoy['center']
                distance = np.sqrt((x - cx)**2 + (y - cy)**2)
                if distance < r + buoy['width'] / 2:
                    is_new = False
                    break
            
            if is_new:
                buoys.append({
                    'center': (int(x), int(y)),
                    'width': int(2 * r),
                    'is_circle': True
                })
        
        # Return the detected buoys and processed images for visualization
        return buoys, {
            'original': original,
            'preprocessed': preprocessed,
            'binary': combined_binary
        }
    
    def draw_results(self, image, buoys):
        """Draw the detected buoys on the image"""
        result = image.copy()
        
        for buoy in buoys:
            center = buoy['center']
            width = buoy['width']
            radius = int(width // 2)
            
            # Draw center and circle
            cv2.circle(result, center, 3, (0, 255, 255), -1)
            cv2.circle(result, center, radius, (0, 255, 0), 2)
            
            # Draw contour if available
            if 'contour' in buoy:
                cv2.drawContours(result, [buoy['contour']], 0, (255, 0, 0), 2)
                
            # Add width label
            cv2.putText(result, f"W: {width}", 
                       (center[0] - 20, center[1] - radius - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
        return result

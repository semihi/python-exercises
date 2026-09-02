import cv2
import numpy as np
from buoy_detector import BuoyDetector
from config import MultipleConfigsHandler

class AdaptiveBuoyDetector:
    """
    Adaptive buoy detector that automatically determines the
    lighting condition and uses the appropriate configuration.
    """
    
    def __init__(self):
        self.config_handler = MultipleConfigsHandler()
        
    def analyze_lighting(self, image):
        """
        Analyze the image to determine the lighting condition
        Returns: 'normal', 'shadow', or 'shiny'
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Compute histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()  # Normalize
        
        # Calculate statistical features
        mean_brightness = np.mean(gray)
        std_dev = np.std(gray)
        
        # Look at dark and bright pixel ratios
        dark_ratio = np.sum(gray < 50) / gray.size
        bright_ratio = np.sum(gray > 200) / gray.size
        
        # Analyze histogram peaks
        peaks = []
        for i in range(1, 255):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > 0.01:
                peaks.append((i, hist[i]))
        
        # Decide based on features
        if dark_ratio > 0.4 or mean_brightness < 80:
            return "shadow"
        elif bright_ratio > 0.3 or mean_brightness > 180 or std_dev > 70:
            return "shiny"
        else:
            return "normal"
    
    def detect_buoys(self, image):
        """
        Detect buoys in the image by automatically determining
        the lighting condition and using the appropriate configuration
        """
        # Determine lighting condition
        lighting = self.analyze_lighting(image)
        print(f"Detected lighting condition: {lighting}")
        
        # Get the appropriate config
        config = self.config_handler.get_config(lighting)
        
        # Create detector with config and detect buoys
        detector = BuoyDetector(config)
        return detector.detect_buoys(image), detector, lighting

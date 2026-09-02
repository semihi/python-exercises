class BuoyDetectorConfig:
    """Configuration parameters for the buoy detector"""
    
    def __init__(self):
        # Preprocessing parameters
        self.use_grayscale = True
        self.use_histogram_equalization = True
        self.use_gaussian_blur = True
        self.gaussian_kernel_size = 9
        self.gaussian_sigma = 2.0
        
        # Adaptive thresholding
        self.use_adaptive_threshold = True
        self.adaptive_block_size = 11
        self.adaptive_c = 6
        
        # Color thresholding parameters
        self.use_hsv_thresholding = True
        # Example HSV ranges for common buoy colors (adjust based on your specific buoys)
        self.hsv_color_ranges = [
            # Red buoys (two ranges because red wraps around in HSV)
            # {'lower': [0, 100, 100], 'upper': [10, 255, 255]},
            # {'lower': [160, 100, 100], 'upper': [180, 255, 255]},
            # # Green buoys
            # {'lower': [40, 50, 50], 'upper': [80, 255, 255]},
            # Yellow buoys
            {'lower': [20, 100, 0], 'upper': [40, 255, 255]},
        ]
        
        self.use_lab_thresholding = False
        self.lab_color_ranges = []  # Configure if needed for specific lighting
        
        # Morphological operations
        self.use_morphological_operations = True
        self.morphological_kernel_size = 5
        self.close_iterations = 2
        self.open_iterations = 1
        
        # Contour filtering
        self.min_contour_area = 1000
        self.max_contour_area = 50000
        self.min_circularity = 0.5
        self.min_aspect_ratio = 0.5
        self.max_aspect_ratio = 2.0
        
        # Hough Circle Transform parameters
        self.use_hough_circles = False
        self.hough_dp = 1.2
        self.hough_min_dist = 20
        self.hough_param1 = 50
        self.hough_param2 = 30
        self.hough_min_radius = 10
        self.hough_max_radius = 100
        
        # Ellipse fitting
        self.use_ellipse_fitting = True
        
        # Canny edge detector parameters
        self.canny_threshold1 = 50
        self.canny_threshold2 = 150


class MultipleConfigsHandler:
    """Handle multiple configurations for different lighting conditions"""
    
    def __init__(self):
        self.normal_config = BuoyDetectorConfig()
        
        self.shadow_config = BuoyDetectorConfig()
        self.shadow_config.use_histogram_equalization = True
        self.shadow_config.adaptive_c = 4  # More aggressive threshold for shadows
        self.shadow_config.hsv_color_ranges = [
            # Adjusted ranges for shadow conditions
            {'lower': [0, 80, 80], 'upper': [10, 255, 255]},
            {'lower': [160, 80, 80], 'upper': [180, 255, 255]},
            {'lower': [40, 40, 40], 'upper': [80, 255, 255]},
            {'lower': [20, 80, 80], 'upper': [40, 255, 255]},
        ]
        
        self.shiny_config = BuoyDetectorConfig()
        self.shiny_config.gaussian_sigma = 2.0  # More blurring for shiny conditions
        self.shiny_config.hsv_color_ranges = [
            # Adjusted ranges for shiny conditions
            {'lower': [0, 100, 150], 'upper': [10, 255, 255]},
            {'lower': [160, 100, 150], 'upper': [180, 255, 255]},
            {'lower': [40, 50, 150], 'upper': [80, 255, 255]},
            {'lower': [20, 100, 150], 'upper': [40, 255, 255]},
        ]
        self.shiny_config.canny_threshold1 = 100  # Higher thresholds for bright conditions
        self.shiny_config.canny_threshold2 = 200
        
    def get_config(self, lighting_condition):
        """Get the appropriate configuration for a lighting condition"""
        if lighting_condition == "shadow":
            return self.shadow_config
        elif lighting_condition == "shiny":
            return self.shiny_config
        else:  # normal
            return self.normal_config

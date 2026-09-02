import cv2
import argparse
import os
import numpy as np
from buoy_detector import BuoyDetector
from config import MultipleConfigsHandler

def parse_arguments():
    parser = argparse.ArgumentParser(description='Detect buoys in images with varying lighting conditions')
    parser.add_argument('--image', type=str, default='10m/left.jpg', 
                       help='Path to the input image (default: 10m/left.jpg)')
    parser.add_argument('--lighting', type=str, default='normal', 
                        choices=['normal', 'shadow', 'shiny'], 
                        help='Lighting condition in the image')
    parser.add_argument('--output', type=str, default=None, help='Path to save the output image')
    parser.add_argument('--show', action='store_true', default=True, 
                       help='Show the results in a window (default: True)')
    parser.add_argument('--no_show', action='store_false', dest='show',
                       help='Do not show the results in a window')
    parser.add_argument('--viz_size', type=int, default=800, help='Maximum size for visualization window')
    parser.add_argument('--screen_fit', action='store_true', help='Scale visualization to fit screen')
    return parser.parse_args()

def get_screen_resolution():
    """Get screen resolution or use a default if not available"""
    try:
        # Try to get screen resolution from OpenCV
        temp_window = "temp"
        cv2.namedWindow(temp_window, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(temp_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        screen_width = cv2.getWindowImageRect(temp_window)[2]
        screen_height = cv2.getWindowImageRect(temp_window)[3]
        cv2.destroyWindow(temp_window)
        
        # Apply some safety margin
        screen_width = int(screen_width * 0.9)  # 90% of screen width
        screen_height = int(screen_height * 0.8)  # 80% of screen height
    except:
        # Default fallback values if can't detect
        screen_width = 1280
        screen_height = 720
    
    return screen_width, screen_height

def main():
    args = parse_arguments()
    
    # Check if image exists
    if not os.path.isfile(args.image):
        print(f"Error: Image file '{args.image}' not found")
        return
    
    # Load image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Failed to load image '{args.image}'")
        return
    
    print(f"Processing image: {args.image} with {args.lighting} lighting condition")
    
    # Get appropriate config for the lighting condition
    config_handler = MultipleConfigsHandler()
    config = config_handler.get_config(args.lighting)
    
    # Create detector with selected config
    detector = BuoyDetector(config)
    
    # Detect buoys
    buoys, processed_images = detector.detect_buoys(image)
    
    # Draw results
    result_image = detector.draw_results(image, buoys)
    
    # Create visualization
    preprocessed = processed_images['preprocessed']
    binary = processed_images['binary']
    
    # Make sure preprocessed is 3-channel for display
    if len(preprocessed.shape) == 2:
        preprocessed = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)
    if len(binary.shape) == 2:
        binary_colored = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    # Print results
    print(f"Detected {len(buoys)} buoys:")
    for i, buoy in enumerate(buoys):
        print(f"Buoy {i+1}: Center={buoy['center']}, Width={buoy['width']}")
    
    # Show results if requested
    if args.show:
        # Get image dimensions
        h, w = image.shape[:2]
        
        # If screen_fit is enabled, adjust viz_size to fit screen
        if args.screen_fit:
            screen_width, screen_height = get_screen_resolution()
            
            # Our visualization will have 2x2 grid, so adjust accordingly
            max_viz_width = screen_width
            max_viz_height = screen_height
            
            # Calculate dimensions that maintain aspect ratio and fit screen
            grid_aspect_ratio = (w * 2) / (h * 2)  # 2x2 grid
            
            if max_viz_width / grid_aspect_ratio <= max_viz_height:
                # Width constrained
                args.viz_size = max_viz_width // 2
            else:
                # Height constrained
                args.viz_size = (max_viz_height * grid_aspect_ratio) // 2
        
        # Calculate the appropriate size for visualization based on viz_size
        aspect_ratio = w / h
        
        # Determine how to resize based on the viz_size parameter
        if w > h:
            new_w = min(w, args.viz_size)  # Limit to original or viz_size
            new_h = int(new_w / aspect_ratio)
        else:
            new_h = min(h, args.viz_size)  # Limit to original or viz_size
            new_w = int(new_h * aspect_ratio)
        
        # Resize all images for visualization
        image_resized = cv2.resize(image, (new_w, new_h))
        preprocessed_resized = cv2.resize(preprocessed, (new_w, new_h))
        binary_colored_resized = cv2.resize(binary_colored, (new_w, new_h))
        result_image_resized = cv2.resize(result_image, (new_w, new_h))
        
        # Create a visualization with original, preprocessed, binary, and result images
        viz = np.zeros((new_h*2, new_w*2, 3), dtype=np.uint8)
        viz[0:new_h, 0:new_w] = image_resized
        viz[0:new_h, new_w:new_w*2] = preprocessed_resized
        viz[new_h:new_h*2, 0:new_w] = binary_colored_resized
        viz[new_h:new_h*2, new_w:new_w*2] = result_image_resized
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.5, min(1.0, new_w / 400))  # Scale font based on image size
        cv2.putText(viz, "Original", (10, 30), font, font_scale, (255, 255, 255), 2)
        cv2.putText(viz, "Preprocessed", (new_w+10, 30), font, font_scale, (255, 255, 255), 2)
        cv2.putText(viz, "Binary", (10, new_h+30), font, font_scale, (255, 255, 255), 2)
        cv2.putText(viz, "Result", (new_w+10, new_h+30), font, font_scale, (255, 255, 255), 2)
        
        print(f"Visualization size: {viz.shape[1]}x{viz.shape[0]} pixels")
        
        # Show the visualization in a window that can be resized by the user if needed
        cv2.namedWindow("Buoy Detection", cv2.WINDOW_NORMAL)
        cv2.imshow("Buoy Detection", viz)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Save output if path is provided
    if args.output:
        cv2.imwrite(args.output, result_image)
        print(f"Result saved to {args.output}")

if __name__ == "__main__":
    main()
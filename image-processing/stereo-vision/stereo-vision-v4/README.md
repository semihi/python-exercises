# Buoy Detection in Varying Lighting Conditions

This project implements buoy detection in images under varying lighting conditions such as shadows, shining/glare, and normal lighting.

## Features

- Robust detection of buoys in different lighting conditions
- Multiple detection methods combined for best results:
  - Color thresholding in HSV/LAB color spaces
  - Adaptive thresholding
  - Contour detection and filtering
  - Hough circle transform
  - Morphological operations
- Automatic lighting condition detection
- Parameter configurations optimized for different lighting scenarios

## Installation

Requirements:
- Python 3.6+
- OpenCV (cv2)
- NumPy

```bash
pip install opencv-python numpy
```

## Usage

### Basic Usage with Manual Lighting Selection

```bash
# Use with defaults (10m/left.jpg, normal lighting, show results)
python main.py

# Or specify parameters
python main.py --image path/to/image.jpg --lighting normal --show --screen_fit
```

Options:
- `--image`: Path to the input image (default: 10m/left.jpg)
- `--lighting`: Lighting condition ['normal', 'shadow', 'shiny'] (default: normal)
- `--output`: Path to save the output image (optional)
- `--show`: Show the results in a window (default: True)
- `--no_show`: Do not show the results in a window
- `--viz_size`: Maximum size for visualization window (default: 800 pixels)
- `--screen_fit`: Automatically scale visualization to fit your screen (optional)

### Adaptive Usage with Automatic Lighting Detection

```bash
# Use with defaults (10m/left.jpg, show results)
python adaptive_main.py

# Or specify parameters
python adaptive_main.py --image path/to/image.jpg --show --screen_fit
```

Options:
- `--image`: Path to the input image (default: 10m/left.jpg)
- `--output`: Path to save the output image (optional)
- `--show`: Show the results in a window (default: True)
- `--no_show`: Do not show the results in a window
- `--viz_size`: Maximum size for visualization window (default: 800 pixels)
- `--screen_fit`: Automatically scale visualization to fit your screen (optional)

## How It Works

1. **Preprocessing**: The image is preprocessed to handle varying lighting conditions through:
   - Grayscale conversion
   - Histogram equalization
   - Gaussian blur
   - Adaptive thresholding

2. **Color Thresholding**: Multiple color spaces (HSV/LAB) are used to create binary masks for buoy detection.

3. **Contour Detection**: Contours are detected from binary masks and filtered based on:
   - Area
   - Circularity
   - Aspect ratio

4. **Circle Detection**: The Hough Circle Transform is used as a supplementary method.

5. **Buoy Property Calculation**: For each detected buoy, properties like center coordinates and width are calculated.

6. **Adaptive Detection**: The system can automatically determine the lighting condition and select the appropriate configuration.

## Customizing Detection

You can modify the parameters in `config.py` to optimize the detection for your specific buoy types and lighting conditions.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- OpenCV documentation and community
- Computer vision principles and techniques

## Troubleshooting

- **False detections**: Adjust the thresholds in `config.py` to better match your buoy's color and size
- **Missed detections**: Lower the thresholds or try different color spaces
- **Incorrect lighting detection**: Modify the logic in `analyze_lighting()` to better suit your environment
- **Display issues**: 
  - If visualization doesn't fit on screen, use the `--screen_fit` option
  - Alternatively, set a smaller `--viz_size` value (e.g., `--viz_size 600`)
  - The window is resizable, so you can manually adjust it as needed

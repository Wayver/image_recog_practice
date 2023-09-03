import cv2
import numpy as np
import argparse

def detect_lines(image_path, rho, theta, threshold_percent):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use Canny edge detection
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    
    # Calculate the threshold for Hough Line Transform
    height, width = edges.shape[:2]
    average_dimension = (height + width) / 2
    threshold = int(threshold_percent * average_dimension / 100)

    # Apply Hough Transform to detect lines
    lines = cv2.HoughLines(edges, rho, theta, threshold)

    # Draw the detected lines on the original image
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return image

if __name__ == "__main__":
    # Create a command-line argument parser
    parser = argparse.ArgumentParser(description='Line Detection in an Image using Hough Transform')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('--rho', type=int, default=1, help='Resolution of the accumulator in pixels (default: 1)')
    parser.add_argument('--theta', type=float, default=np.pi / 180, help='Angle resolution of the accumulator in radians (default: 1 degree)')
    parser.add_argument('--threshold_percent', type=float, default=60, help='Threshold as a percentage of the average image dimension (default: 60)')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Detect lines in the provided image
    result_image = detect_lines(args.image_path, args.rho, args.theta, args.threshold_percent)

    # Display the image with detected lines
    cv2.imshow('Detected Lines', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


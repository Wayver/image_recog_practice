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

    # Create a list to store intersection points
    intersection_points = []

    # Draw the detected lines in red
    if lines is not None:
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                rho1, theta1 = lines[i][0]
                rho2, theta2 = lines[j][0]

                # Calculate intersection point (if lines are not parallel)
                if np.abs(theta1 - theta2) > 1e-5:
                    A = np.array([[np.cos(theta1), np.sin(theta1)],
                                  [np.cos(theta2), np.sin(theta2)]])
                    B = np.array([rho1, rho2])
                    intersection_point = np.linalg.solve(A, B)
                    intersection_point = tuple(map(int, intersection_point))
                    
                    # Add the intersection point to the list
                    intersection_points.append(intersection_point)

                # Draw lines on the image in red
                a = np.cos(theta1)
                b = np.sin(theta1)
                x0 = a * rho1
                y0 = b * rho1
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

                a = np.cos(theta2)
                b = np.sin(theta2)
                x0 = a * rho2
                y0 = b * rho2
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Mark intersection points on the image in blue
    for point in intersection_points:
        cv2.circle(image, point, 5, (255, 0, 0), -1)  # Mark as blue

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

    # Detect lines and intersections in the provided image
    result_image = detect_lines(args.image_path, args.rho, args.theta, args.threshold_percent)

    # Display the image with detected lines and marked intersection points
    cv2.imshow('Detected Lines', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

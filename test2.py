import cv2
import numpy as np

# Load the image
img = cv2.imread('im1.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection
edges = cv2.Canny(gray, 200, 300)

# Find the contours of the object's edges
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour
largest_contour = max(contours, key=cv2.contourArea)

# Find the top and bottom points of the contour
topmost = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
bottommost = tuple(largest_contour[largest_contour[:, :, 1].argmax()][0])

# Calculate the height of the object in pixels
height_pixels = bottommost[1] - topmost[1]
print('The height of the object in pixels is:', height_pixels)

# Define the pixel density or resolution of the fingerprint image (pixels per centimeter)
pixel_density = 100  # Example value, adjust according to your image and requirements

# Calculate the height of the object in centimeters
height_cm = height_pixels / pixel_density
print('The height of the object in centimeters is:', height_cm, 'cm')

# Calculate the width of the object in pixels
leftmost = tuple(largest_contour[largest_contour[:, :, 0].argmin()][0])
rightmost = tuple(largest_contour[largest_contour[:, :, 0].argmax()][0])
width_pixels = rightmost[0] - leftmost[0]
print('The width of the object in pixels is:', width_pixels)

# Calculate the width of the object in centimeters
width_cm = width_pixels / pixel_density
print('The width of the object in centimeters is:', width_cm, 'cm')

# Apply thresholding in the gray image to create a binary image
ret, thresh = cv2.threshold(gray, 150, 255, 0)

# Find the contours using the binary image
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print("Number of contours in the image:", len(contours))

# Find the contour with the largest area
largest_contour = max(contours, key=cv2.contourArea)

# Compute the area and perimeter of the contour
area = cv2.contourArea(largest_contour)
perimeter = cv2.arcLength(largest_contour, True)
perimeter = round(perimeter, 4)
print('Area:', area)
print('Perimeter:', perimeter)

# Draw the contour on the image
cv2.drawContours(img, [largest_contour], -1, (0, 255, 255), 3)
x1, y1 = largest_contour[0, 0]
cv2.putText(img, f'Area: {area}', (x1, y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
cv2.putText(img, f'Perimeter: {perimeter}', (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# Display the image
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

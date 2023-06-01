import cv2
import numpy as np

# open source image file
image = cv2.imread('2__F_Left_little_finger.BMP', cv2.IMREAD_UNCHANGED)

# Check the number of channels in the image
if len(image.shape) == 2:  # Single-channel image
    image_gray = image
elif len(image.shape) == 3:  # Three-channel image
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
else:
    raise ValueError("Unsupported image format")



# Apply thresholding to create a binary image where the fingerprint area is white
_, threshold =  cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Apply a morphological operation to close small gaps in the fingerprint area and create a solid blob
kernel = np.ones((5, 5), np.uint8)
closed = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
#closed = cv2.dilate(closed, kernel,iterations = 1)

h, w = closed .shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)
cv2.floodFill(closed , mask, (0,0), 0)

# Invert the closed image to make the fingerprint area black and the background white
inverted = cv2.bitwise_not(closed)


# onvert image to blck and white
thresh, image_edges = cv2.threshold(inverted, 100, 255, cv2.THRESH_BINARY)

# create canvas
canvas = np.zeros(inverted.shape, np.uint8)
canvas.fill(255)

# create background mask
mask = np.zeros(inverted.shape, np.uint8)
mask.fill(255)


# get all contours
contours_draw, hierachy = cv2.findContours(image_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# get most significant contours
contours_mask, hierachy = cv2.findContours(image_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# draw all contours
cv2.drawContours(canvas, contours_mask, -1, (255, 255, 255), cv2.FILLED)



# contours traversal
for contour in range(len(contours_draw)):
    # draw current contour
    cv2.drawContours(canvas, contours_draw, contour, (0, 0, 0), 3)



# most significant contours traversal
for contour in range(len(contours_mask)):
    # create mask
    if contour != 1:
        cv2.fillConvexPoly(mask, contours_mask[contour], (0, 0, 0))
# display the image in a window
cv2.imshow('Original', image)
cv2.imshow('Contours', canvas)
cv2.imshow('Contours', closed)

cv2.imwrite('finger1.bmp',closed)

# Load the image (assuming the image is called "blob.png")
image = cv2.imread('finger1.bmp', 0)  # Load as grayscale

# Preprocess the image (e.g., thresholding)
_, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get the largest contour
contour = max(contours, key=cv2.contourArea)

# Fit an ellipse
ellipse = cv2.fitEllipse(contour)

# Extract the parameters of the ellipse
center, axes, angle = ellipse
length = max(axes)
width = min(axes)

# Draw lines representing length and width
image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
cv2.ellipse(image_color, ellipse, (0, 255, 0), 2)  # Draw green ellipse

# Calculate the endpoints for the length line
angle_rad = np.deg2rad(angle)
cos_angle = np.cos(angle_rad)
sin_angle = np.sin(angle_rad)
half_length = length / 2
pt1_length = (int(center[0] - sin_angle * half_length), int(center[1] + cos_angle * half_length))
pt2_length = (int(center[0] + sin_angle * half_length), int(center[1] - cos_angle * half_length))
cv2.line(image_color, pt1_length, pt2_length, (0, 0, 255), 2)  # Draw red length line

# Calculate the endpoints for the width line
half_width = width / 2
pt1_width = (int(center[0] - cos_angle * half_width), int(center[1] - sin_angle * half_width))
pt2_width = (int(center[0] + cos_angle * half_width), int(center[1] + sin_angle * half_width))
cv2.line(image_color, pt1_width, pt2_width, (0, 0, 255), 2)  # Draw red width line

# Print the length and width
print("Length: {:.2f}".format(length/96))
print("Width: {:.2f}".format(width/96))

# Display the image
cv2.imshow("Blob with lines", image_color)
cv2.imwrite('final.png', image_color)
cv2.waitKey(0)
cv2.destroyAllWindows()





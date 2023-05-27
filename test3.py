import cv2

# Load the image
img = cv2.imread('im1.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply thresholding in the gray image to create a binary image
ret, thresh = cv2.threshold(gray, 150, 255, 0)

# Find the contours using the binary image
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour
largest_contour = max(contours, key=cv2.contourArea)

# Find the bounding rectangle of the contour
x, y, width, height = cv2.boundingRect(largest_contour)

# Retrieve the image dimensions in inches
width_inch = 1  # Example value, replace with the actual width in inches
height_inch = .7  # Example value, replace with the actual height in inches

# Retrieve the DPI information from the image metadata (if available)
dpi = None  # Example value, replace with the actual DPI value from metadata if available

# Calculate the pixel density
pixel_density = dpi if dpi is not None else max(img.shape[1] / width_inch, img.shape[0] / height_inch)

# Calculate the height of the fingerprint in inches
height_inch = height / pixel_density
print('Height:', height_inch, 'in')

# Calculate the width of the fingerprint in inches
width_inch = width / pixel_density
print('Width:', width_inch, 'in')

# Draw the bounding rectangle on the image
cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 2)

# Display the image
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Load the image
img = cv2.imread('image/img3.png')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Apply Canny edge detection
# edges = cv2.Canny(gray, 50, 200)
#
# # Find the contours of the object's edges
# contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# # Find the largest contour
# largest_contour = max(contours, key=cv2.contourArea)
#
# # Find the top and bottom points of the contour
# topmost = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
# bottommost = tuple(largest_contour[largest_contour[:, :, 1].argmax()][0])
#
# # Calculate the height of the object
# height = bottommost[1] - topmost[1]
#
# print('The height of the object is:', height)
#
# # Convert the height from pixels to centimeters
# pixel_density = 100 # Assume the camera has a pixel density of 100 PPCM
# height_cm = height / pixel_density
#
# print('The height of the object in cm is:', height_cm, 'cm')
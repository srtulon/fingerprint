import cv2
import numpy as np

# Load the image
img = cv2.imread('im1.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

 # Apply Canny edge detection
edges = cv2.Canny(gray, 200, 300)

#Find the contours of the object's edges
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#Find the largest contour
largest_contour = max(contours, key=cv2.contourArea)

 #Find the top and bottom points of the contour
topmost = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
bottommost = tuple(largest_contour[largest_contour[:, :, 1].argmax()][0])
#Calculate the height of the object
height = bottommost[1] - topmost[1]

print('The height of the object is:', height)
#
# # Convert the height from pixels to centimeters
# pixel_density = 100 # Assume the camera has a pixel density of 100 PPCM
# height_cm = height / pixel_density
#
# print('The height of the object in cm is:', height_cm, 'cm')


# Apply thresholding in the gray image to create a binary image
ret,thresh = cv2.threshold(gray,150,255,0)

# Find the contours using binary image
contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
print("Number of contours in image:",len(contours))
cnt = contours[0]

# compute the area and perimeter
area = cv2.contourArea(cnt)
perimeter = cv2.arcLength(cnt, True)
perimeter = round(perimeter, 4)
print('Area:', area)
print('Perimeter:', perimeter)
img1 = cv2.drawContours(img, [cnt], -1, (0,255,255), 3)
x1, y1 = cnt[0,0]
cv2.putText(img1, f'Area:{area}', (x1, y1+40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
cv2.putText(img1, f'Perimeter:{perimeter}', (x1, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

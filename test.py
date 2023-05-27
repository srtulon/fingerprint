import cv2

def compute_fingerprint_dimensions(image_path):
    # Load the fingerprint image
    fingerprint_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Binarize the image using thresholding
    _, binarized_image = cv2.threshold(fingerprint_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find the contour of the fingerprint
    contours, _ = cv2.findContours(binarized_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the bounding rectangle around the fingerprint contour
    x, y, w, h = cv2.boundingRect(contours[0])

    return w, h

# Path to the fingerprint image
image_path = 'im1.jpg'

# Compute the dimensions of the fingerprint
width, height = compute_fingerprint_dimensions(image_path)

print("Width:", width)
print("Height:", height)

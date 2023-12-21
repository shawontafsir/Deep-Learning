import cv2
import numpy as np

# Initialize variables
points = []
image = None


# Function to handle mouse events
def draw_polygon(event, x, y, flags, param):
    global points, image

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))


# Create an OpenCV window for displaying images
cv2.namedWindow("Select Bounding Points", cv2.WINDOW_NORMAL)

# Load the image
image_path = "image3.jpg"
image = cv2.imread(image_path)

# Create a clone of the original image to draw on
clone = image.copy()

# Resize the image for display (adjust the dimensions as needed)
cv2.resizeWindow("Select Bounding Points", image.shape[1], image.shape[0])

# Set the mouse callback function
cv2.setMouseCallback("Select Bounding Points", draw_polygon)

while True:
    # Show the image with bounding points
    for point in points:
        cv2.circle(clone, point, 5, (0, 0, 255), -1)  # Draw red circles at the selected points
    cv2.imshow("Select Bounding Points", clone)

    key = cv2.waitKey(1) & 0xFF

    # Clear the bounding points when 'c' key is pressed
    if key == ord("c"):
        points = []
        clone = image.copy()

    # Exit the program when 'q' key is pressed
    elif key == ord("q"):
        break

# Create a binary mask based on the selected bounding points
mask = np.zeros(image.shape[:2], dtype=np.uint8)
if len(points) == 4:
    pts = np.array(points, np.int32)
    cv2.fillPoly(mask, [pts], (255, 255, 255))

# Save the binary mask to a file or use it as needed
cv2.imwrite("img2.png", mask)

# Close all OpenCV windows
cv2.destroyAllWindows()

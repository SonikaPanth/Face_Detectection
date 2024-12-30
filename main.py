import cv2

# Load Haar Cascade for face detection
a = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Start video capture
b = cv2.VideoCapture(0)

if not b.isOpened():
    print("Error: Could not access the camera.")
    exit()

while True:
    # Capture each frame
    c_rec, d_image = b.read()
    if not c_rec:
        print("Error: Could not read frame.")
        break

    # Convert to grayscale for face detection
    e = cv2.cvtColor(d_image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    f = a.detectMultiScale(e, 1.3, 6)

    # Draw rectangles around detected faces
    for (x1, y1, w1, h1) in f:
        cv2.rectangle(d_image, (x1, y1), (x1 + w1, y1 + h1), (255, 0, 0), 5)

    # Display the resulting frame
    cv2.imshow('Face Detection', d_image)

    # Exit on pressing 'Esc' key
    h = cv2.waitKey(1) & 0xFF
    if h == 27:  # Escape key
        break

# Release resources
b.release()
cv2.destroyAllWindows()

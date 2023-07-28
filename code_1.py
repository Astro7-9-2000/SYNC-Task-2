
import cv2

# Load the Haar cascade xml file for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect faces and draw rectangles around them
def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

        # Check if the face is masked or not based on mouth area
        mouth_area = gray[y+h//2:y+h, x:x+w]
        _, mouth_mask = cv2.threshold(mouth_area, 100, 255, cv2.THRESH_BINARY)

        # Calculate a dynamic threshold based on the area of the face
        threshold = (w * h) // 150

        # Tag the face as MASKED or UNMASKED based on mouth mask
        tag = "MASKED" if cv2.countNonZero(mouth_mask) > threshold else "UNMASKED"

        # Display the tag above the face rectangle
        cv2.putText(frame, tag, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame from the webcam
    ret, frame = cap.read()

    # Detect faces in the frame
    frame = detect_faces(frame)

    # Display the frame
    cv2.imshow('Face Mask Detection', frame)

    # Check for the 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the windows
cap.release()
cv2.destroyAllWindows()


import cv2

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the video capture (webcam 0 is typically the default)
video_capture = cv2.VideoCapture(0)

# Check if the webcam opened correctly
if not video_capture.isOpened():
    print("Error: Could not open video capture.")
    exit()

while True:
    # Capture a frame from the webcam
    success, frame = video_capture.read()

    # Check if the frame was captured successfully
    if not success:
        print("Failed to capture frame. Exiting...")
        break

    # Convert the frame to grayscale (required for face detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # If a face is detected, simulate login and exit
    if len(faces) > 0:
        cv2.putText(frame, "Face Detected - User Logged In", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        print("User logged in successfully!")  # Simulate user login action
        cv2.imshow("Live Face Detection", frame)  # Show the frame with the face detected
        cv2.waitKey(3000)  # Keep the window open for 3 seconds before closing
        break  # Exit after user logs in

    # Show the live feed
    cv2.imshow("Live Face Detection", frame)

    # Check if the user presses the 'Esc' key to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the video capture object and close any OpenCV windows
video_capture.release()
cv2.destroyAllWindows()  # Close all OpenCV windows

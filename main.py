import dlib
import cv2

# Load the pre-trained face recognition model
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Load a sample image for face recognition
known_image = cv2.imread("known_face.jpeg")
known_image = cv2.cvtColor(known_image, cv2.COLOR_BGR2RGB)
faces = detector(known_image, 1)

if len(faces) > 0:
    shape = sp(known_image, faces[0])
    known_face_encoding = facerec.compute_face_descriptor(known_image, shape)

# Open a video capture stream
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()
    
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = detector(gray)
    
    # Recognize faces
    for face in faces:
        shape = sp(frame, face)
    face_encoding = facerec.compute_face_descriptor(frame, shape)
    
    # Compare with known face encoding
    if 'known_face_encoding' in locals() and len(faces) > 0:
        # Use NumPy for face distance calculation
        import numpy as np
        match = np.linalg.norm(np.array(face_encoding) - np.array(known_face_encoding))
    
        if match < 0.6 :  # Tune Threshold (A greater value means more strict recognition)
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
            cv2.putText(frame, "Known Face", (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 2)
            cv2.putText(frame, "Unknown Face", (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the frame with recognized faces
    cv2.imshow('Face Recognition', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()

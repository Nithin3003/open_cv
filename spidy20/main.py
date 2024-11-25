import cv2

# Error handling: Check if OpenCV is installed
if not cv2.getBuildInformation() is None:
    print("OpenCV successfully imported!")
else:
    print("Error: OpenCV not found. Please install OpenCV using pip install opencv-python")
    exit()

# Error handling: Check if 'cars.xml' exists
if not cv2.CascadeClassifier('cars.xml').empty():
    print("Car cascade classifier loaded successfully!")
else:
    print("Error: 'cars.xml' file not found or invalid. Please download a pre-trained car detection model.")
    exit()

cap = cv2.VideoCapture('Video.mp4')

# Error handling: Check if video file is opened correctly
if cap.isOpened():
    print("Video file opened successfully!")
else:
    print("Error: Could not open video file. Please check the file path.")
    exit()

car_cascade = cv2.CascadeClassifier('cars.xml')
car_count = 0  # Initialize car count

while True:
    ret, frames = cap.read()

    # Error handling: Check if frame is read successfully
    if not ret:
        print("Error: Could not read video frame. Exiting...")
        break

    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.1, 9)

    for (x, y, w, h) in cars:
        car_count += 1  # Increment car count for each detected car
        plate = frames[y:y + h, x:x + w]
        cv2.rectangle(frames, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(frames, (x, y - 40), (x + w, y), (0, 255, 0), -2)
        cv2.putText(frames, 'Car', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Optional: Display the detected car region of interest (ROI) in a separate window
        cv2.imshow('Detected Car', plate)

    # Display the video frame with car detections and car count
    # lab1 = "Car Count: " + str(car_count)
    # cv2.putText(frames, lab1, (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (147, 20, 255), 3)
    frames = cv2.resize(frames, (600, 400))  # Resize the frame for better display
    cv2.imshow('Car Detection System', frames)

    k = cv2.waitKey(30) & 0xff
    if k == 27:  # 'Esc' key pressed to exit
        break

cap.release()
cv2.destroyAllWindows()
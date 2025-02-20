import cv2

# Function to perform face detection
def faceBox(net, frame, conf_threshold=0.7):
    if frame is None:
        return None, []  # Return None if the frame is None

    frameDnn = frame.copy()
    frameHeight = frameDnn.shape[0]
    frameWidth = frameDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()

    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameDnn, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return frameDnn, bboxes

# Load network
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)
faceNet = cv2.dnn.readNet(faceModel, faceProto)

# Open a video capture object for the default camera (0)
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()

    # Break the loop if there is an issue reading the frame
    if not ret:
        print("Error reading the frame.")
        break

    frameFace, bboxes = faceBox(faceNet, frame)

    for bbox in bboxes:
        # code for further processing and display
        face = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = ["Male", "Female"][genderPreds[0].argmax()]

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)'][agePreds[0].argmax()]

        label = "{},{}".format(gender, age)
        cv2.rectangle(frameFace, (bbox[0], bbox[1]-30), (bbox[2], bbox[1]), (0, 255, 0), -1)
        cv2.putText(frameFace, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow("Age-Gender", frameFace)

    # Check for user input to exit the loop
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

# Release the video capture object and close all windows
video.release()
cv2.destroyAllWindows()

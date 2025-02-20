import cv2
import os

def face_box(net, frame, conf_threshold=0.7):
    if frame is None:
        return None, []

    frame_dnn = frame.copy()
    frame_height, frame_width = frame_dnn.shape[:2]
    blob = cv2.dnn.blobFromImage(frame_dnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()

    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1, y1, x2, y2 = map(int, detections[0, 0, i, 3:7] * [frame_width, frame_height, frame_width, frame_height])
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame_dnn, (x1, y1), (x2, y2), (0, 255, 0), 1)

    return frame_dnn, bboxes

# File paths for models
age_proto, age_model = "age_deploy.prototxt", "age_net.caffemodel"
gender_proto, gender_model = "gender_deploy.prototxt", "gender_net.caffemodel"
emotion_proto, emotion_model = "emotion_model.prototxt", "emotion_model.caffemodel"
face_proto, face_model = "opencv_face_detector.pbtxt", "opencv_face_detector_uint8.pb"

# Check if emotion model files exist
emotion_files_exist = all(os.path.isfile(file_path) for file_path in [emotion_proto, emotion_model])

if not emotion_files_exist:
    print("Error: Emotion model files not found.")
else:
    # Load networks if files exist
    age_net = cv2.dnn.readNet(age_model, age_proto)
    gender_net = cv2.dnn.readNet(gender_model, gender_proto)
    emotion_net = cv2.dnn.readNet(emotion_model, emotion_proto)
    face_net = cv2.dnn.readNet(face_model, face_proto)

    # Open a video capture object for the default camera (0)
    video = cv2.VideoCapture(0)

    while True:
        ret, frame = video.read()

        if not ret:
            print("Error reading the frame.")
            break

        frame_face, bounding_boxes = face_box(face_net, frame)

        for bbox in bounding_boxes:
            face_roi = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = ["Male", "Female"][gender_preds[0].argmax()]

            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)'][age_preds[0].argmax()]

            emotion_net.setInput(blob)
            emotion_preds = emotion_net.forward()
            emotion = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'][emotion_preds[0].argmax()]

            label = "{}, {}, {}".format(gender, age, emotion)
            cv2.rectangle(frame_face, (bbox[0], bbox[1] - 30), (bbox[2], bbox[1]), (0, 255, 0), -1)
            cv2.putText(frame_face, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Age-Gender-Emotion", frame_face)

        k = cv2.waitKey(1)
        if k == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

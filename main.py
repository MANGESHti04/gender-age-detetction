import cv2
import argparse
import logging
from typing import List, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define paths for the model files
FACE_DETECTION_MODEL = "face_detection.pb"
AGE_GENDER_MODEL = "age_gender_model.pb"
AGE_LABELS_FILE = "age_labels.txt"
GENDER_LABELS_FILE = "gender_labels.txt"


class FaceDetection:
    def __init__(self, model_path: str):
        self.model = cv2.dnn.readNetFromTensorflow(model_path)
        self.input_size = (300, 300)

    def detect_faces(self, frame: cv2.Mat) -> List[Tuple[int, int, int, int]]:
        """
        Detects faces in the given frame and returns their coordinates.

        Args:
            frame: Input video frame

        Returns:
            A list of tuples (x1, y1, x2, y2) representing face coordinates.
        """
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, self.input_size, (104.0, 177.0, 123.0))
        self.model.setInput(blob)
        output = self.model.forward()

        faces = []
        for i in range(output.shape[2]):
            confidence = output[0, 0, i, 2]
            if confidence > 0.5:
                x1 = int(output[0, 0, i, 3] * frame.shape[1])
                y1 = int(output[0, 0, i, 4] * frame.shape[0])
                x2 = int(output[0, 0, i, 5] * frame.shape[1])
                y2 = int(output[0, 0, i, 6] * frame.shape[0])
                faces.append((x1, y1, x2, y2))
        return faces


class AgeGenderPrediction:
    def __init__(self, model_path: str, labels_file: str):
        self.model = cv2.dnn.readNetFromTensorflow(model_path)
        with open(labels_file, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]

    def predict(self, face_frame: cv2.Mat) -> Tuple[str, int]:
        """
        Predicts age and gender from a face image.

        Args:
            face_frame: Input face image

        Returns:
            A tuple (gender, age).
        """
        blob = cv2.dnn.blobFromImage(
            face_frame, 1.0, (200, 200), (104.0, 177.0, 123.0))
        self.model.setInput(blob)
        output = self.model.forward()

        gender_index = 0
        age_index = 1

        # Get the index of the highest confidence for gender
        max_gender_confidence = -1
        best_gender_index = 0
        for i in range(output[gender_index].shape[1]):
            if output[gender_index, 0, i] > max_gender_confidence:
                max_gender_confidence = output[gender_index, 0, i]
                best_gender_index = i

        gender_label = self.labels[best_gender_index]

        # Get the age prediction
        age_label = int(output[age_index].argmax())

        return (gender_label, age_label)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Face detection and age-gender prediction')
    parser.add_argument('--video', type=str, default='',
                        help='Path to the video file (default: webcam)')
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='Minimum confidence for face detection (default: 0.5)')
    args = parser.parse_args()

    # Initialize face detector
    logger.info("Initializing face detection model...")
    face_detector = FaceDetection(FACE_DETECTION_MODEL)

    # Initialize age-gender prediction model
    logger.info("Initializing age and gender prediction models...")
    ag_predictor = AgeGenderPrediction(AGE_GENDER_MODEL, AGE_LABELS_FILE)

    # Set confidence threshold from command line argument
    face_detector.confidence = args.confidence

    # Open video capture
    if args.video:
        logger.info(f"Opening video file: {args.video}")
        cap = cv2.VideoCapture(args.video)
    else:
        logger.info("Using webcam...")
        cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to read frame")
            break

        # Detect faces
        faces = face_detector.detect_faces(frame)

        for (x1, y1, x2, y2) in faces:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Extract face
            face_frame = frame[y1:y2, x1:x2]

            try:
                gender, age = ag_predictor.predict(face_frame)
                label = f"{gender}, {age}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            except Exception as e:
                logger.error(f"Error predicting age and gender: {str(e)}")
                pass

        # Display the frame
        cv2.imshow('Face Detection and Age-Gender Prediction', frame)

        # Break if ESC is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

import argparse
import os
import cv2
import pyrootutils
from torchvision import transforms

from commons.face_age_module import FaceAgeModule, Nets

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)


class RealTimePredictor:
    def __init__(self, ckpt_paths):
        self.models = [
            FaceAgeModule.load_from_checkpoint(
                path, net=Nets.PretrainedEfficientNet, normalize_age_by=80
            )
            for path in ckpt_paths
        ]

        for model in self.models:
            model.eval()
            model.freeze()

        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def predict(self, img):
        img = self.transforms(img)
        img = img.reshape(1, 3, 224, 224).cuda()

        predictions = [model.forward(img) for model in self.models]

        prediction = sum(predictions) / len(self.models)
        prediction = prediction * 80
        prediction = prediction.clip(1, 80)

        return prediction.item()


def parse_args():
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "-c", "--checkpoints_dir", help="Directory of checkpoints to load"
    )

    args = argParser.parse_args()

    dir = args.checkpoints_dir
    ckpt_paths = [os.path.join(dir, x) for x in os.listdir(dir)]

    return ckpt_paths


def put_bounding_box_with_prediction(video_frame, predictor, face_classifier):
    gray_image = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for x, y, w, h in faces:
        frame = cv2.rectangle(video_frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

        face = video_frame[y : y + h, x : x + w]
        prediction = predictor.predict(face)
        cv2.putText(
            frame,
            str(round(prediction, ndigits=2)),
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )


def main():
    (ckpt_paths) = parse_args()

    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    video_capture = cv2.VideoCapture(0)
    predictor = RealTimePredictor(ckpt_paths=ckpt_paths)

    while True:
        result, video_frame = video_capture.read()
        if result is False:
            break

        put_bounding_box_with_prediction(video_frame, predictor, face_classifier)
        cv2.imshow("Face recognition", video_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

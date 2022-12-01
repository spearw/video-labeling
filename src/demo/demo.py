import torch
import numpy as np
import cv2
import pafy
import validators
import os
from time import time


class ObjectDetection:

    def __init__(self):

        self.model = self.load_model()
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def load_model(self):

        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def score_frame(self, frame):

        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
        return labels, cord

    def class_to_label(self, x):

        return self.classes[int(x)]

    def plot_boxes(self, results, frame):

        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame

    def detect_source_type(self, source):

        if validators.url(source):
            return "url"
        elif os.path.isfile(source):
           return "file"
        else:
            return False

    def get_video_from_url(self, source):

        play = pafy.new(source).streams[-1]
        assert play is not None
        self.out_file = play.title + ".mp4"
        return cv2.VideoCapture(play.url)

    def get_cap(self, source):

        source_type = self.detect_source_type(source)
        if source_type == "url":
            cap = self.get_video_from_url(source)
        elif source_type == "file":
            cap = cv2.VideoCapture(source)
            self.out_file = source + "-labeled.mp4"
        else:
            ##TODO: handle error
            pass
        return cap


    def detect(self, source):

        cap = self.get_cap(source)

        assert cap.isOpened()
        x_shape = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        y_shape = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        four_cc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(self.out_file, four_cc, 60, (x_shape, y_shape))
        total_fps = []
        while cap.isOpened():
            start_time = time()
            ret, frame = cap.read()

            if ret == True:

                # Write boxes on frame
                results = self.score_frame(frame)
                frame = self.plot_boxes(results, frame)

                # Calculate speed
                end_time = time()
                fps = 1 / np.round(end_time - start_time, 3)
                total_fps.append(fps)

                # Write frames to file
                out.write(frame)

                # Display the resulting frame
                cv2.imshow('Frame', frame)

                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

                # Break the loop
            else:
                break


        print(f"Average FPS: {np.mean(total_fps)}")
        cap.release()

# Create a new object and execute.
detector = ObjectDetection()
#detector.detect("https://www.youtube.com/watch?v=2oWosA91i2g&t=1s")
#detector.detect("https://www.youtube.com/watch?v=qxnBcZ0CETg")
detector.detect("https://www.youtube.com/watch?v=xgFmk6PXZkg")
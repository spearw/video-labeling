# core analysis
import cv2
import torch
import numpy as np

# youtube handler
import pafy

# input verification
import os
import validators

# benchmarking
from time import time


class ObjectDetection:
    """Creates and saves a labeled object-detection video from a source
    """

    def __init__(self):

        # load model
        self.model = self.load_model()

        # load cuda if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        """Load the model yolov5 into memory

        Returns:
            model: the pretrained yolov5 model
        """

        model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
        return model

    def model_frame(self, frame):
        """Applies loaded model to frame

        Args:
            frame (numpy.ndarray): the array representation of the frame. Built from opencv.videocapture.read()

        Returns:
            labels (numpy.ndarray): array containing labels for each object detected
            coord (numpy.ndarray): array containing coordinated for each object detected
        """

        # apply model
        self.model.to(self.device)

        # extract desired results
        frame = [frame]
        results = self.model(frame)
        labels, coord = (
            results.xyxyn[0][:, -1].numpy(),
            results.xyxyn[0][:, :-1].numpy(),
        )
        return labels, coord

    def draw_boxes(self, results, frame):
        """Draws boxes in frame with given coordinates and labels

        Args:
            results (tuple(numpy.ndarray, numpy.ndarray)): labels and coords from score_frame.
            frame (numpy.ndarray): the array representation of the frame. Built from opencv.videocapture.read()

        Returns:
            frame (numpy.ndarray): the array representation of the frame with boxes and labels drawn on it
        """

        labels, coord = results
        x_shape, y_shape = frame.shape[1], frame.shape[0]

        # iterate over detected object
        for i in range(len(labels)):
            row = coord[i]

            # plot if certainty score over threshold (score is percentile 0-1)
            if row[4] >= 0.3:

                # draw rectangle
                x1, y1, x2, y2 = (
                    int(row[0] * x_shape),
                    int(row[1] * y_shape),
                    int(row[2] * x_shape),
                    int(row[3] * y_shape),
                )
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 1)

                # label object
                cv2.putText(
                    frame,
                    self.model.names[
                        int(labels[i])
                    ],  # Look up label name from label index
                    (x1, y1),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.9,
                    bgr,
                    1,
                )

        return frame

    def detect_source_type(self, source):
        """Detects the source input

        Args:
            source (str): input source

        Returns:
            str: str representation of source type
        """

        if validators.url(source):
            return "url"
        elif os.path.isfile(source):
            return "file"
        else:
            return "unknown"

    def load_youtube_video(self, source):
        """loads youtube video from youtube source url + names self.out_file_name

        Args:
            source (str): input source

        Returns:
            cv2.VideoCapture: cv2 object of source video
        """

        play = pafy.new(source).streams[-1]
        assert play is not None
        self.out_file_name = play.title + ".mp4"
        return cv2.VideoCapture(play.url)

    def load_local_video(self, source):
        """loads local video from file source + names self.out_file_name

        Args:
            source (str): input source

        Returns:
            cv2.VideoCapture: cv2 object of source video
        """
        name, ext = os.path.splitext(source)
        self.out_file_name = "{name}-labeled{ext}".format(name=name, ext=ext)
        return cv2.VideoCapture(source)

    def get_video_capture(self, source):
        """loads cv2.videocapture from various sources

        Args:
            source (str): input source

        Returns:
            cv2.VideoCapture: cv2 object of source video
        """

        source_type = self.detect_source_type(source)

        ##TODO: handle other types
        if source_type == "url":
            video_capture = self.load_youtube_video(source)
        elif source_type == "file":
            video_capture = self.load_local_video(source)
        else:
            raise Exception("Invalid source type")
        return video_capture

    def detect(self, source):
        """Creates and saves a labeled object-detection video from a source

        Args:
            source (str): input source. Supports youtube urls and file sources
        """

        video_capture = self.get_video_capture(source)

        assert video_capture.isOpened()

        # get input shape to match for output shape
        x_shape = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        y_shape = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # set output format
        four_cc = cv2.VideoWriter_fourcc(*"mp4v")

        # create video writer
        out = cv2.VideoWriter(self.out_file_name, four_cc, 60, (x_shape, y_shape))

        total_fps = []

        while video_capture.isOpened():

            # Init time for benchmarking
            start_time = time()
            ret, frame = video_capture.read()

            if ret == True:

                # Write boxes on frame
                results = self.model_frame(frame)
                frame = self.draw_boxes(results, frame)

                # Calculate speed
                end_time = time()
                fps = 1 / np.round(end_time - start_time, 3)
                total_fps.append(fps)

                # Write frames to file
                out.write(frame)

                # Display the resulting frame
                cv2.imshow("Frame", frame)

                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    break

            # break when capture is closed
            else:
                break

        # print benchmark
        print(f"Average FPS: {np.mean(total_fps)}")

        # clean up
        video_capture.release()


# Create a new object and execute.
detector = ObjectDetection()
detector.detect("https://www.youtube.com/watch?v=xgFmk6PXZkg")

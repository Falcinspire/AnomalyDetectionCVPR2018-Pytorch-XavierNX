import argparse
import logging
import sys
from typing import List

import numpy as np
import torch
from torch import Tensor, nn
from PyQt5.QtCore import Qt, QUrl  # pylint: disable=no-name-in-module
from PyQt5.QtGui import QIcon, QPalette, QPixmap, QImage  # pylint: disable=no-name-in-module
from PyQt5.QtWidgets import (  # pylint: disable=no-name-in-module
    QApplication,
    QWidget,
    QPushButton,
    QLabel,
    QStyle,
    QSizePolicy,
    QFileDialog,
    QProgressBar,
    QGridLayout,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from tqdm import tqdm
import cv2
import time
from numpy.lib.function_base import copy
import threading
from pyqtgraph import PlotWidget, plot
import pyqtgraph
from multiprocessing import Process, Pipe, Array, Lock

from data_loader import SingleVideoIter
from feature_extractor import to_segments
from utils.load_model import load_models
from utils.utils import build_transforms


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video Demo For Anomaly Detection")

    parser.add_argument(
        "--feature_extractor",
        help="path to the 3d model for feature extraction",
    )
    parser.add_argument(
        "--feature_method",
        default="c3d",
        choices=["c3d", "mfnet", "r3d101", "r3d101"],
        help="method to use for feature extraction",
    )
    parser.add_argument(
        "--ad_model", 
        help="path to the trained AD model",
    )
    parser.add_argument(
        "--mock_delay", 
        type=int, 
        help="Use a mock delay for the model instead of running it. Inference is always 0.",
    )
    parser.add_argument(
        "--use-torch-trt", 
        action='store_true', 
        help="Optimize the model with torch2trt",
    )

    return parser.parse_args()

CLIP_LENGTH = 16

class ModelInference:
    def __init__(self, feature_extractor, ad_model, feature_method, should_use_torch_trt):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        anomaly_detector, feature_extractor = load_models(
            feature_extractor,
            ad_model,
            features_method=feature_method,
            device=device,
            use_trt=should_use_torch_trt,
        )

        transforms = build_transforms(mode=feature_method)

        # Warmup the models. If this is ignored, the initial part of the video 
        # will not be processed as the warmup occurs there. Not sure why the 
        # models need to warm up.
        feature_extractor(torch.zeros((1, 3, 16, 112, 112)).to(device))
        anomaly_detector(torch.zeros((1, 4096)).to(device))

        self.device = device
        self.transforms = transforms
        self.feature_extractor = feature_extractor
        self.anomaly_detector = anomaly_detector

    def predict(self, clip):
        clip = self.transforms(clip).unsqueeze(0)
        outputs = self.feature_extractor(clip.to(self.device)).detach().cpu().numpy()
        outputs = to_segments(outputs, 1)
        return self.anomaly_detector(torch.tensor(outputs).to(self.device)).item()

class MockDelayInference:
    def __init__(self, delay):
        self.delay = delay
    def predict(self, clip):
        accurate_sleep(self.delay)
        return 0

def inference_process(pipe: Pipe, feature_extractor, ad_model, feature_method, forced_delay, should_use_torch_trt):
    print('loading models...')
    inference_machine = \
        ModelInference(feature_extractor, ad_model, feature_method, should_use_torch_trt) \
        if forced_delay is None else \
        MockDelayInference(forced_delay)

    print('inference process ready...')
    pipe.send(('status', 'ready'))

    frame_buffer = []
    frame_buffer_numbers = []
    frame_buffer_dirty = False
    cap = None

    last_frame_number = -1
    with torch.no_grad():
        while True:
            while pipe.poll():
                message = pipe.recv()
                if message[0] == 'open':
                    frame_buffer = []
                    frame_buffer_numbers = []
                    frame_buffer_dirty = False

                    filepath = message[1]
                    cap = cv2.VideoCapture(filepath)
                elif message[0] == 'frame':
                    frame_number = message[1]
                    assert len(frame_buffer_numbers) == 0 or frame_number == frame_buffer_numbers[-1] + 1
                    frame_buffer_numbers.append(frame_number)
                    frame_buffer_numbers = frame_buffer_numbers[-CLIP_LENGTH:]
                    frame = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB)
                    frame_buffer.append(frame)
                    frame_buffer = frame_buffer[-CLIP_LENGTH:]
                    frame_buffer_dirty = True
                elif message[0] == 'terminate':
                    return
            
            if not frame_buffer_dirty:
                continue

            if len(frame_buffer) < CLIP_LENGTH:
                continue
            
            last_frame_number = frame_buffer_numbers[-1]
            clip = torch.tensor(copy(frame_buffer[-CLIP_LENGTH:]))
            new_pred = inference_machine.predict(clip)
            frame_buffer_dirty = False
            
            pipe.send(('inference', (last_frame_number - 16 + 1, last_frame_number, new_pred)))

def accurate_sleep(seconds):
    wait_until = time.perf_counter() + seconds
    while True:
        now = time.perf_counter()
        if now >= wait_until: 
            return
class Window(QWidget):
    """
    Anomaly detection gui
    Based on media player code from:
    https://codeloop.org/python-how-to-create-media-player-in-pyqt5/
    """

    def __init__(self, pipe) -> None:
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.frame_number = 0
        self.frame_times = []
        self.pred_x_buffer = []
        self.pred_y_buffer = []
        self.pipe = pipe

        self.setWindowTitle("Anomaly Media Player")
        self.setGeometry(350, 100, 700, 500)
        self.setWindowIcon(QIcon("player.png"))

        p = self.palette()
        p.setColor(QPalette.Window, Qt.black)
        self.setPalette(p)

        self.init_ui()

        self.show()

    def init_ui(self):
        # create open button
        self.openBtn = QPushButton("Open Video")
        self.openBtn.clicked.connect(self.open_file)

        # create label
        self.label = QLabel()

        self.fps_label = QLabel()
        self.fps_label.setText(' --fps inference')
        self.fps_label.setStyleSheet("QLabel { background-color : white; color : black; }");

        self.playback_fps_label = QLabel()
        self.playback_fps_label.setText(' --fps playback')
        self.playback_fps_label.setStyleSheet("QLabel { background-color : white; color : black; }");

        self.skipped_label = QLabel()
        self.skipped_label.setText(' Skipped -- individual frames')
        self.skipped_label.setStyleSheet("QLabel { background-color : white; color : black; }");

        # create grid layout
        gridLayout = QGridLayout()

        # AD signal
        self.graphWidget = PlotWidget()
        self.graphWidget.setYRange(-0.1, 1.1)
        self.graphData = self.graphWidget.plot([], [])

        # set widgets to the hbox layout
        gridLayout.addWidget(self.graphWidget, 0, 0, 1, 5)
        gridLayout.addWidget(self.label, 1, 0, 5, 5)
        gridLayout.addWidget(self.openBtn, 6, 0, 1, 1)
        gridLayout.addWidget(self.fps_label, 6, 1, 1, 1)
        gridLayout.addWidget(self.playback_fps_label, 6, 2, 1, 1)
        gridLayout.addWidget(self.skipped_label, 6, 3, 1, 1)

        self.setLayout(gridLayout)

    def open_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Video")
        if filename != "":
            self.openBtn.setDisabled(True)
            self.frame_number = 0
            self.frame_times = []
            self.pred_x_buffer = []
            self.pred_y_buffer = []

            while self.pipe.poll():
                self.pipe.recv() # discard leftover messages from last video
            self.pipe.send(('open', filename))

            self.play_video(filename)

    def play_video(self, filepath):
        cap = cv2.VideoCapture(filepath)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        spf = 1 / fps
        last_frame_time = -1

        inference_fps = None
        rolling_playback_fps = []

        total_clips_skipped = 0
        total_frames_skipped = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pxmap = QPixmap.fromImage(QImage(frame, frame.shape[1],frame.shape[0],frame.strides[0],QImage.Format_RGB888))

            delta = time.perf_counter() - last_frame_time
            accurate_sleep(max(0, spf - delta))

            # poll events after sleep
            if self.pipe.poll():
                message = self.pipe.recv()
                if message[0] == 'inference':
                    start_frame, end_frame = message[1][0], message[1][1]
                    prediction = message[1][2]

                    initial_frame_time = self.frame_times[start_frame]
                    inference_fps = (end_frame - start_frame + 1) / (time.perf_counter() - initial_frame_time)

                    if len(self.pred_y_buffer) > 0:
                        last_frame_predicted = self.pred_x_buffer[-1]
                        total_clips_skipped += end_frame - (last_frame_predicted + 1)
                        total_frames_skipped += max(0, start_frame - (last_frame_predicted + 1))

                    self.pred_x_buffer.append(end_frame)
                    self.pred_x_buffer = self.pred_x_buffer[-64:]
                    self.pred_y_buffer.append(prediction)
                    self.pred_y_buffer = self.pred_y_buffer[-64:]

            self.label.setPixmap(pxmap)
            cur_frame_time = time.perf_counter()
            rolling_playback_fps.append(1/(cur_frame_time - last_frame_time))
            self.frame_times.append(cur_frame_time)
            last_frame_time = cur_frame_time

            self.pipe.send(('frame', self.frame_number))
            self.frame_number += 1

            self.skipped_label.setText(f' Skipped {total_frames_skipped} individual frames')

            if len(rolling_playback_fps) == 10:
                self.playback_fps_label.setText(f' {np.mean(rolling_playback_fps):.0f}fps playback')
                rolling_playback_fps = []

            if inference_fps is not None:
                self.fps_label.setText(f' {inference_fps:.0f}fps inference')

            if self.frame_number % 10 == 0:
                self.graphData.setData(self.pred_x_buffer, self.pred_y_buffer)

            QApplication.processEvents()
        cap.release()
        self.openBtn.setDisabled(False)

if __name__ == "__main__":
    args = get_args()

    print('starting model process...')
    parent_conn, child_conn = Pipe()
    p = Process(target=inference_process, args=(child_conn, args.feature_extractor, args.ad_model, args.feature_method, args.mock_delay, args.use_torch_trt))
    p.start()

    await_status_message = parent_conn.recv()
    if await_status_message[0] != 'status':
        raise Exception(f'Expected status message, got {await_status_message[0]}')
    if await_status_message[1] != 'ready':
        raise Exception(f'Expected inference process to be "ready", got {await_status_message[1]}')

    print('starting app...')
    app = QApplication(sys.argv)
    window = Window(parent_conn)
    rc = app.exec_()
    print('terminating model process...')
    parent_conn.send(('terminate',))
    p.join()
    sys.exit(rc)

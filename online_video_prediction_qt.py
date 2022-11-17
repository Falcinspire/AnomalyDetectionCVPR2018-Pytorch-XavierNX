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

from data_loader import SingleVideoIter
from feature_extractor import to_segments
from utils.load_model import load_models
from utils.utils import build_transforms


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video Demo For Anomaly Detection")

    parser.add_argument(
        "--feature_extractor",
        required=True,
        help="path to the 3d model for feature extraction",
    )
    parser.add_argument(
        "--feature_method",
        default="c3d",
        choices=["c3d", "mfnet", "r3d101", "r3d101"],
        help="method to use for feature extraction",
    )
    parser.add_argument(
        "--ad_model", required=True, help="path to the trained AD model"
    )
    parser.add_argument(
        "--n_segments",
        type=int,
        default=32,
        help="number of segments to use for features averaging",
    )

    return parser.parse_args()

clip_length = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

global_edit_lock = threading.Condition()
frame_buffer = []
frame_times = []
frame_number = -1
pred_x = []
pred_y = []
processing_fps = 0
close_processing_thread = False

def processing_thread(transforms, feature_extractor, anomaly_detector):
    global clip_length
    global device
    global frame_buffer
    global frame_number
    global pred_x
    global pred_y
    global processing_fps
    global close_processing_thread

    last_frame_number = -1
    with torch.no_grad():
        while True:
            # clip_processing_start = time.perf_counter()
            if close_processing_thread:
                break
            with global_edit_lock:
                if last_frame_number == frame_number or len(frame_buffer) < clip_length:
                    continue
            
            with global_edit_lock:
                last_frame_number = frame_number
                clip = torch.tensor(copy(frame_buffer[-clip_length:]))
                clip_start_time = frame_times[-clip_length]
            clip = transforms(clip).unsqueeze(0)
            outputs = feature_extractor(clip.to(device)).detach().cpu().numpy()
            outputs = to_segments(outputs, 1)
            new_pred = anomaly_detector(torch.tensor(outputs).to(device))
            with global_edit_lock:
                clip_processing_time = time.perf_counter() - clip_start_time
                pred_x.append(last_frame_number)
                pred_y.append(new_pred.item())
                processing_fps = clip_length / clip_processing_time

class Window(QWidget):
    """
    Anomaly detection gui
    Based on media player code from:
    https://codeloop.org/python-how-to-create-media-player-in-pyqt5/
    """

    def __init__(self) -> None:
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        openBtn = QPushButton("Open Video")
        openBtn.clicked.connect(self.open_file)

        # create label
        self.label = QLabel()
        self.fps_label = QLabel()
        self.fps_label.setText(' -- fps')
        self.fps_label.setStyleSheet("QLabel { background-color : white; color : black; }");

        # create grid layout
        gridLayout = QGridLayout()

        # AD signal
        self.graphWidget = PlotWidget()
        self.graphWidget.setYRange(-0.1, 1.1)
        self.graphData = self.graphWidget.plot([], [])

        # set widgets to the hbox layout
        gridLayout.addWidget(self.graphWidget, 0, 0, 1, 5)
        gridLayout.addWidget(self.label, 1, 0, 5, 5)
        gridLayout.addWidget(openBtn, 6, 0, 1, 1)
        gridLayout.addWidget(self.fps_label, 6, 1, 1, 1)

        self.setLayout(gridLayout)

    def open_file(self):
        global frame_buffer
        global frame_times
        global frame_number
        global pred_x
        global pred_y

        filename, _ = QFileDialog.getOpenFileName(self, "Open Video")
        if filename != "":
            pred_x = []
            pred_y = []
            frame_buffer = []
            frame_times = []
            frame_number = -1

            self.play_video(filename)

    def play_video(self, filepath):
        global frame_buffer
        global frame_times
        global frame_number
        global pred_x
        global pred_y

        cap = cv2.VideoCapture(filepath)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        spf = 1 / fps
        last_frame_time = -1
        MAGIC_DELAY = 0.01 # extra delay that isn't captured by time.perf_counter()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                return
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pxmap = QPixmap.fromImage(QImage(frame, frame.shape[1],frame.shape[0],frame.strides[0],QImage.Format_RGB888))
            delta = time.perf_counter() - last_frame_time
            time.sleep(max(0, spf - delta - MAGIC_DELAY))
            with global_edit_lock:
                frame_number += 1
                frame_buffer.append(frame)
                last_frame_time = time.perf_counter()
                frame_times.append(last_frame_time)
                frame_buffer = frame_buffer[-clip_length:]
                self.label.setPixmap(pxmap)
                self.fps_label.setText(f' {processing_fps:.0f} fps')

                if len(pred_x) % 10 == 0:
                    self.graphData.setData(pred_x[-64:], pred_y[-64:])

            QApplication.processEvents()
        cap.release()

if __name__ == "__main__":
    args = get_args()

    print('loading models...')
    anomaly_detector, feature_extractor = load_models(
        args.feature_extractor,
        args.ad_model,
        features_method=args.feature_method,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    transforms = build_transforms(mode=args.feature_method)

    print('warming up models...')
    # Warmup the models. If this is ignored, the initial part of the video 
    # will not be processed as the warmup occurs there. Not sure why the 
    # models need to warm up.
    feature_extractor(torch.zeros((1, 3, 16, 112, 112)).to(device))
    anomaly_detector(torch.zeros((1, 4096)).to(device))

    print('starting processing thread...')
    threading.Thread(target=processing_thread, args=(transforms, feature_extractor, anomaly_detector)).start()

    print('starting app...')
    app = QApplication(sys.argv)
    window = Window()
    rc = app.exec_()
    close_processing_thread = True
    sys.exit(rc)

import argparse

from numpy.lib.function_base import copy
import torch
import cv2
import time
import threading

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
    parser.add_argument(
        "--video",
        required=True,
        help="path to the video",
    )

    return parser.parse_args()

frame_buffer_resource = threading.Condition()
video_over = False
frame_buffer = []
frame_number = -1

def video_thread(video_path, buffer_length):
    global video_over
    global frame_buffer
    global frame_number

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    spf = 1 / fps
    last_frame_time = -1
    MAGIC_DELAY_MS = 0.01 # remove extra delay to account for... calculating the delay and updating the display, I guess?

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            frame_buffer_resource.acquire()
            video_over = True
            frame_buffer_resource.release()
            break
        delta = time.perf_counter() - last_frame_time
        time.sleep(max(0, spf - delta - MAGIC_DELAY_MS))
        cv2.imshow('video', frame)
        last_frame_time = time.perf_counter()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            frame_buffer_resource.acquire()
            video_over = True
            frame_buffer_resource.release()
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_buffer_resource.acquire()
        frame_buffer.append(frame)
        frame_buffer = frame_buffer[-buffer_length:]
        frame_number += 1
        frame_buffer_resource.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    args = get_args()

    anomaly_detector, feature_extractor = load_models(
        args.feature_extractor,
        args.ad_model,
        features_method=args.feature_method,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )


    transforms = build_transforms(mode=args.feature_method)

    clip_length = 16
    video_path = args.video
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Warmup the models. If this is ignored, the initial part of the video 
    # will not be processed as the warmup occurs there. Not sure why the 
    # models need to warm up.
    feature_extractor(torch.zeros((1, 3, 16, 112, 112)).to(device))
    anomaly_detector(torch.zeros((1, 4096)).to(device))
    
    threading.Thread(target=video_thread, args=(video_path, clip_length)).start()
    
    with torch.no_grad():
        while True:
            frame_buffer_resource.acquire()
            cur_frame_number = frame_number
            is_video_over = video_over
            clip = torch.tensor(copy(frame_buffer))
            frame_buffer_resource.release()

            if is_video_over:
                break
            if clip.shape[0] == clip_length:
                clip = transforms(clip).unsqueeze(0)
                outputs = feature_extractor(clip.to(device)).detach().cpu().numpy()
                outputs = to_segments(outputs, 1)
                new_pred = anomaly_detector(torch.tensor(outputs).to(device))
                print(cur_frame_number, new_pred.item())
    
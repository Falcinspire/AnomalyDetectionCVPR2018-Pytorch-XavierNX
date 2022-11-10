import argparse

from numpy.lib.function_base import copy
import torch
import cv2

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
        "--video",
        required=True,
        help="path to the video",
    )

    return parser.parse_args()

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
    
    cap = cv2.VideoCapture(video_path)
    frame_buffer = []
    with torch.no_grad():
        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow('video', frame.copy())
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_buffer.append(frame)
            frame_buffer = frame_buffer[-clip_length:]
            if len(frame_buffer) == clip_length:
                clip = torch.tensor(copy(frame_buffer))
                clip = transforms(clip).unsqueeze(0)
                outputs = feature_extractor(clip.to(device)).detach().cpu().numpy()
                outputs = to_segments(outputs, 1)
                new_pred = anomaly_detector(torch.tensor(outputs).to(device))
                print(new_pred.item())
    cap.release()
    cv2.destroyAllWindows()

    
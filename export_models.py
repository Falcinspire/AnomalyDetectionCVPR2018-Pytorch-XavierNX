import torch
import torch.onnx
import argparse

from utils.load_model import load_models

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

    return parser.parse_args()

def export_model(model, input_shape, model_name):
    data = torch.zeros(input_shape, requires_grad=True)
    torch.onnx.export(
        model,
        data,
        f"{model_name}.onnx",
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names = ['input'],
        output_names = ['output'],
        dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}}
    )

if __name__ == "__main__":
    args = get_args()

    anomaly_detector, feature_extractor = load_models(
        args.feature_extractor,
        args.ad_model,
        features_method=args.feature_method,
        device="cpu",
    )

    export_model(feature_extractor, (1, 3, 16, 112, 112), 'feature_extractor')
    export_model(anomaly_detector, (1, 4096), 'anomaly_detector')
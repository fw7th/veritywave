import argparse

from predictor import predict


def main():
    parser = argparse.ArgumentParser(description="AI vs Real Image/Video Detector")
    parser.add_argument("--input", required=True, help="Path to image or video file")
    parser.add_argument(
        "--model_dir", default="models", help="Path to saved model directory"
    )
    args = parser.parse_args()

    predict(args.input, model_dir=args.model_dir)


if __name__ == "__main__":
    main()

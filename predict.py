# predict.py

import argparse
import json
from model_utils import load_checkpoint, predict


def main():
    parser = argparse.ArgumentParser(description="Predict the class of an input image")
    parser.add_argument("input_image", help="Path to the input image")
    parser.add_argument("checkpoint", help="Path to the checkpoint file")
    parser.add_argument(
        "--top_k",
        dest="top_k",
        type=int,
        default=1,
        help="Return top K most likely classes",
    )
    parser.add_argument(
        "--category_names",
        dest="category_names",
        help="Mapping of categories to real names",
    )
    parser.add_argument(
        "--gpu", dest="gpu", action="store_true", help="Use GPU for inference"
    )

    args = parser.parse_args()

    model = load_checkpoint(args.checkpoint)

    probs, classes = predict(args.input_image, model, args.top_k, args.gpu)

    if args.category_names:
        with open(args.category_names, "r") as f:
            cat_to_name = json.load(f)
        classes = [cat_to_name[class_] for class_ in classes]

    for prob, class_ in zip(probs, classes):
        print(f"{class_}: {prob*100:.2f}%")


if __name__ == "__main__":
    main()

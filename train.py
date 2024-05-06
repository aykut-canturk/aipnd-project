# train.py

import argparse
from datetime import datetime
import torch
from model_utils import build_model, train_model, validate_model, save_checkpoint
from data_utils import load_data


def main():
    parser = argparse.ArgumentParser(description="Train a neural network on a dataset")
    parser.add_argument("data_dir", help="Path to the directory containing the data")
    parser.add_argument(
        "--save_dir",
        dest="save_dir",
        default="checkpoints/",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--arch",
        dest="arch",
        default="vgg16",
        choices=["vgg13", "vgg16", "resnet18"],
        help="Architecture (vgg16 or resnet18)",
    )
    parser.add_argument(
        "--learning_rate",
        dest="learning_rate",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--hidden_units",
        dest="hidden_units",
        type=int,
        default=512,
        help="Number of hidden units",
    )
    parser.add_argument(
        "--epochs", dest="epochs", type=int, default=20, help="Number of epochs"
    )
    parser.add_argument(
        "--gpu", dest="gpu", action="store_true", help="Use GPU for training"
    )

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    print(f"Device: {device}")

    start_time = datetime.now()

    train_loader, valid_loader, _, class_to_idx = load_data(args.data_dir)
    model = build_model(device, args.arch, args.hidden_units)

    train_model(model, device, train_loader, args.learning_rate, args.epochs)
    validate_model(model, device, valid_loader)
    save_checkpoint(model, class_to_idx, args.arch, args.save_dir)

    print(f"Total training time: {datetime.now() - start_time}")


if __name__ == "__main__":
    main()

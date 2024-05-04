import argparse
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim


def main():
    parser = argparse.ArgumentParser(description="Train a new network on a data set.")
    parser.add_argument("data_directory", type=str, help="Data directory")
    parser.add_argument(
        "--save_dir", type=str, default="./", help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="vgg16",
        help="Architecture [available: vgg16, vgg13]",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.01, help="Learning rate"
    )
    parser.add_argument(
        "--hidden_units", type=int, default=512, help="Number of hidden units"
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")

    args = parser.parse_args()

    # Load data from args.data_directory
    transform = transforms.Compose(
        [transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor()]
    )
    dataset = datasets.ImageFolder(args.data_directory, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    print(f"Data loaded from {args.data_directory}")

    # Create model using args.arch
    if args.arch == "vgg16":
        model = models.vgg16(pretrained=True)
    elif args.arch == "vgg13":
        model = models.vgg13(pretrained=True)
    else:
        print(f"Unknown architecture {args.arch}")
        return

    print(f"Model architecture: {args.arch}")

    # Freeze the pre-trained parameters
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final layer
    num_ftrs = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, args.hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(args.hidden_units, 102),
        nn.LogSoftmax(dim=1),
    )

    print(f"Model classifier replaced with: {model.classifier}")

    # Set up criterion and optimizer using args.learning_rate and args.hidden_units
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.classifier.parameters(), lr=args.learning_rate, momentum=0.9
    )

    print(f"Criterion: {criterion}")
    print(f"Optimizer: {optimizer}")

    # If args.gpu is True, use a GPU for training
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    model = model.to(device)

    print(f"Device set to {device}")

    # Train model for args.epochs epochs
    for epoch in range(args.epochs):
        print(f"Starting epoch {epoch+1}/{args.epochs}")
        running_loss = 0
        i = 0
        for inputs, labels in dataloader:
            # print i/n for every steps
            i += 1
            print(f"Step {i}/{len(dataloader)}")
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(
            f"Epoch {epoch+1} completed. Training loss: {running_loss/len(dataloader)}"
        )

    # Save trained model to args.save_dir
    torch.save(model.state_dict(), f"{args.save_dir}/checkpoint.pth")
    print(f"Model saved to {args.save_dir}/checkpoint.pth")


if __name__ == "__main__":
    main()

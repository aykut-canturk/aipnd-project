import argparse
import torch
from torchvision import datasets, transforms, models

def main():
    parser = argparse.ArgumentParser(description='Train a new network on a data set.')
    parser.add_argument('data_directory', type=str, help='Data directory')
    parser.add_argument('--save_dir', type=str, help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg16', help='Architecture [available: vgg16]')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')

    args = parser.parse_args()

    # Load data from args.data_directory
    transform = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor()])
    dataset = datasets.ImageFolder(args.data_directory, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    # Create model using args.arch
    model = models.__dict__[args.arch](pretrained=True)

    # Set up criterion and optimizer using args.learning_rate and args.hidden_units
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train model for args.epochs epochs
    for epoch in range(args.epochs):
        for inputs, labels in dataloader:
            if args.gpu:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

    # If args.gpu is True, use a GPU for training
    if args.gpu:
        model = model.to('cuda')

    # Save trained model to args.save_dir
    torch.save(model.state_dict(), args.save_dir)

if __name__ == "__main__":
    main()
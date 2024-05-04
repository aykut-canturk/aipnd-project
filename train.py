import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('--save_dir', default='./')
    parser.add_argument('--arch', default='vgg16')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--hidden_units', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--gpu', action='store_true')

    args = parser.parse_args()

    print(args)
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(device)
    # Load data
    data_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor()])
    image_datasets = datasets.ImageFolder(args.data_dir, transform=data_transforms)
    print(image_datasets)
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=64, shuffle=True)
    print(dataloaders)
    # Define model
    if args.arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        print("Unsupported architecture")

    # Set hyperparameters
    model.classifier = nn.Sequential(nn.Linear(25088, args.hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(args.hidden_units, 102),
                                     nn.LogSoftmax(dim=1))

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    print(model)
    # Train model
    model.to(device)
    print("Training model...")
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}.. ")
        running_loss = 0
        i = 0
        for inputs, labels in dataloaders:
            print(f"Batch {i+1}/{len(dataloaders)}.. ")
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            i += 1

        print(f"Epoch {epoch+1}/{args.epochs}.. "
              f"Train loss: {running_loss/len(dataloaders):.3f}.. ")

    # Save checkpoint
    checkpoint = {'arch': args.arch,
                  'classifier': model.classifier,
                  'state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'class_to_idx': image_datasets.class_to_idx}
    print("Saving checkpoint...")
    torch.save(checkpoint, args.save_dir + '/checkpoint.pth')
    print("Checkpoint saved.")
if __name__ == "__main__":
    main()

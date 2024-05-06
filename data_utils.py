import os
import torch
from torchvision import datasets, transforms
import numpy as np
from PIL import Image


def load_data(data_dir):
    train_dir = os.path.join(data_dir, "train")
    valid_dir = os.path.join(data_dir, "valid")
    test_dir = os.path.join(data_dir, "test")

    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "valid": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    image_datasets = {
        "train": datasets.ImageFolder(train_dir, transform=data_transforms["train"]),
        "valid": datasets.ImageFolder(valid_dir, transform=data_transforms["valid"]),
        "test": datasets.ImageFolder(test_dir, transform=data_transforms["test"]),
    }
    dataloaders = {
        "train": torch.utils.data.DataLoader(
            image_datasets["train"], batch_size=64, shuffle=True
        ),
        "valid": torch.utils.data.DataLoader(image_datasets["valid"], batch_size=32),
        "test": torch.utils.data.DataLoader(image_datasets["test"], batch_size=32),
    }
    return (
        dataloaders["train"],
        dataloaders["valid"],
        dataloaders["test"],
        image_datasets["train"].class_to_idx,
    )


def process_image(image_path):
    image = Image.open(image_path)
    # Resize the image where the shortest side is 256 pixels
    width, height = image.size
    aspect_ratio = width / height
    if width < height:
        new_width = 256
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = 256
        new_width = int(new_height * aspect_ratio)

    image = image.resize((new_width, new_height))

    # Center crop the image to 224x224
    left = (new_width - 224) / 2
    top = (new_height - 224) / 2
    right = (new_width + 224) / 2
    bottom = (new_height + 224) / 2

    image = image.crop((left, top, right, bottom))

    # Convert image to numpy array
    np_image = np.array(image) / 255.0

    # Normalize the image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    # Reorder dimensions
    np_image = np_image.transpose((2, 0, 1))

    return np_image

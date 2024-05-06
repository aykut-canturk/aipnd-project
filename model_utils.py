import os
from datetime import datetime
import zipfile
import numpy as np
import torch
from torch import nn
from torchvision.models import VGG16_Weights, VGG13_Weights, ResNet18_Weights
from torchvision import models
from PIL import Image


def __get_model(arch):
    if arch == "resnet18":
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    elif arch == "vgg16":
        model = models.vgg16(weights=VGG16_Weights.DEFAULT)
    elif arch == "vgg13":
        model = models.vgg13(weights=VGG13_Weights.DEFAULT)
    else:
        print(arch)
        raise ValueError("Unsupported model name")
    return model


def build_model(device, arch="vgg16", hidden_units=512):
    model = __get_model(arch)

    if arch == "vgg16":
        input_size = model.classifier[0].in_features
    elif arch == "vgg13":
        input_size = model.classifier[0].in_features
    elif arch == "resnet18":
        input_size = model.fc.in_features
    else:
        print(arch)
        raise ValueError("Invalid architecture. Choose between 'vgg16' and 'resnet18'.")

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # Replace the classifier with a new one
    classifier = nn.Sequential(
        nn.Linear(input_size, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1),
    )

    if arch in ["vgg16", "vgg13"]:
        model.classifier = classifier
    elif arch == "resnet18":
        model.fc = classifier

    model = model.to(device)
    print("Model loaded")
    return model


def train_model(model, device, train_loader, learning_rate=0.001, epochs=1):
    criterion = nn.NLLLoss()

    if isinstance(model, models.ResNet):
        optimizer = torch.optim.SGD(
            model.fc.parameters(), lr=learning_rate, momentum=0.9
        )
    elif isinstance(model, models.VGG):
        optimizer = torch.optim.SGD(
            model.classifier.parameters(), lr=learning_rate, momentum=0.9
        )
    else:
        raise ValueError("Unsupported architecture")

    # Train the model
    train_count = len(train_loader)
    for epoch_idx in range(epochs):
        print(f"Epoch {epoch_idx+1}/{epochs}")
        train_idx = 1
        for inputs, labels in train_loader:
            print(f"train {train_idx}/{train_count} (epoch {epoch_idx+1}/{epochs})")
            train_idx += 1
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print(f"Training loss: {loss.item():.4f}")

    print("Training complete")


def validate_model(model, device, valid_loader):
    criterion = nn.NLLLoss()

    # Initialize the test loss and accuracy
    test_loss = 0
    accuracy = 0

    # Switch the model to evaluation mode
    model.eval()

    with torch.no_grad():
        i = 1
        for inputs, labels in valid_loader:
            print(f"validate {i}/{len(valid_loader)}")
            i += 1
            inputs = inputs.to(device)
            labels = labels.to(device)

            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            _, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(
        f"Test loss: {test_loss/len(valid_loader):.3f}.. "
        f"Test accuracy: {accuracy/len(valid_loader):.3f}"
    )


def save_checkpoint(model, class_to_idx, arch, save_dir="checkpoints/"):
    print(f"Saving checkpoint to {save_dir}")

    # Create a checkpoint dictionary
    checkpoint = {
        "arch": arch,
        "class_to_idx": class_to_idx,
        "state_dict": model.state_dict(),
    }

    if arch == "resnet18":
        checkpoint["fc"] = model.fc
    elif arch in ["vgg16", "vgg13"]:
        checkpoint["classifier"] = model.classifier
    else:
        raise ValueError("Unsupported architecture")

    # Save the checkpoint
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_path = os.path.join(save_dir, f"{arch}.pth")
    __backup_checkpoint(file_path)
    torch.save(checkpoint, file_path)
    print(f"Checkpoint saved to {file_path}")


def __backup_checkpoint(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file {checkpoint_path} does not exist")
        return

    zip_file_name = f"{checkpoint_path}.{datetime.now().strftime('%Y%m%d%H%M%S')}.zip"
    print(f"Backing up existing checkpoint to {zip_file_name}")
    with zipfile.ZipFile(zip_file_name, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(checkpoint_path, arcname=os.path.basename(checkpoint_path))
    print("Backup complete")
    # remove the existing checkpoint
    os.remove(checkpoint_path)


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    arch = checkpoint["arch"]

    model = __get_model(arch)
    model.class_to_idx = checkpoint["class_to_idx"]

    if arch == "resnet18":
        model.fc = checkpoint["fc"]
    elif arch in ["vgg16", "vgg13"]:
        model.classifier = checkpoint["classifier"]

    model.load_state_dict(checkpoint["state_dict"])

    return model


def process_image(image):
    """Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns an Numpy array
    """

    # Open the image

    img = Image.open(image)

    # Resize
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))

    # Crop
    left_margin = (img.width - 224) / 2
    bottom_margin = (img.height - 224) / 2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224

    img = img.crop((left_margin, bottom_margin, right_margin, top_margin))

    # Normalize
    img = np.array(img) / 255
    mean = np.array([0.485, 0.456, 0.406])  # provided mean
    std = np.array([0.229, 0.224, 0.225])  # provided std
    img = (img - mean) / std

    # Move color channels to first dimension as expected by PyTorch
    img = img.transpose((2, 0, 1))

    return img


def predict(image_path, model, topk=5, use_gpu=False):
    # Determine the device and move the model to the correct device
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    model.to(device)

    # Process image
    img = process_image(image_path)

    # Numpy -> Tensor
    image_tensor = torch.from_numpy(img).type(torch.FloatTensor)

    # Move the input data to the correct device
    image_tensor = image_tensor.to(device)

    # Add batch of size 1 to image
    model_input = image_tensor.unsqueeze(0)

    # Probs
    probs = torch.exp(model.forward(model_input))

    # Move probabilities to cpu for numpy operations
    probs = probs.cpu()

    # Top probs
    top_probs, top_labs = probs.topk(topk)
    top_probs = top_probs.detach().numpy().tolist()[0]
    top_labs = top_labs.detach().numpy().tolist()[0]

    # Convert indices to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    # print(f"top_labs: {top_labs}")
    # print(f"idx_to_class: {idx_to_class}")
    top_labels = [idx_to_class[lab] for lab in top_labs if lab in idx_to_class]
    # top_flowers = [cat_to_name[idx_to_class[lab]] for lab in top_labs if lab in idx_to_class]
    # print(f"top_flowers: {top_flowers}")
    return top_probs, top_labels

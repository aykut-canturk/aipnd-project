# model_utils.py
import torch
from torch import nn
from torchvision import models
from PIL import Image
import numpy as np


def build_model(device, arch="vgg16", hidden_units=512):
    if arch == "vgg16":
        model = models.vgg16(pretrained=True)
        input_size = model.classifier[0].in_features
    elif arch == "vgg13":
        model = models.vgg13(pretrained=True)
        input_size = model.classifier[0].in_features
    elif arch == "resnet18":
        model = models.resnet18(pretrained=True)
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
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(
        model.classifier.parameters(), lr=learning_rate, momentum=0.9
    )

    # Train the model
    train_count = len(train_loader)
    i = 0
    for epoch_idx in range(epochs):
        print("Epoch {}/{}".format(epoch_idx, epochs))
        i = 0
        for inputs, labels in train_loader:
            print("Batch {}/{}".format(i, train_count))
            i += 1
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print("Training loss: {:.4f}".format(loss.item()))

    print("Training complete")

    # criterion = nn.NLLLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # device = "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
    # model.to(device)

    # for epoch in range(epochs):
    #     train_loss = 0.0
    #     valid_loss = 0.0
    #     accuracy = 0.0

    #     # Training the model
    #     model.train()
    #     for images, labels in train_loader:
    #         images, labels = images.to(device), labels.to(device)

    #         optimizer.zero_grad()
    #         output = model(images)
    #         loss = criterion(output, labels)
    #         loss.backward()
    #         optimizer.step()

    #         train_loss += loss.item() * images.size(0)

    #     # Validating the model
    #     model.eval()
    #     with torch.no_grad():
    #         for images, labels in valid_loader:
    #             images, labels = images.to(device), labels.to(device)

    #             output = model(images)
    #             loss = criterion(output, labels)

    #             valid_loss += loss.item() * images.size(0)

    #             ps = torch.exp(output)
    #             top_p, top_class = ps.topk(1, dim=1)
    #             equals = top_class == labels.view(*top_class.shape)
    #             accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    #     train_loss = train_loss / len(train_loader.dataset)
    #     valid_loss = valid_loss / len(valid_loader.dataset)
    #     accuracy = accuracy / len(valid_loader)

    #     print(
    #         f"Epoch {epoch+1}/{epochs}.. "
    #         f"Train loss: {train_loss:.3f}.. "
    #         f"Validation loss: {valid_loss:.3f}.. "
    #         f"Validation accuracy: {accuracy:.3f}"
    #     )


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
            print("Batch {}/{}".format(i, len(valid_loader)))
            i += 1
            inputs = inputs.to(device)
            labels = labels.to(device)

            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(
        f"Test loss: {test_loss/len(valid_loader):.3f}.. "
        f"Test accuracy: {accuracy/len(valid_loader):.3f}"
    )


def save_checkpoint(model, class_to_idx, save_dir="checkpoints/"):
    print(f"Saving checkpoint to {save_dir}")
    # checkpoint = {
    #     "arch": arch,
    #     "hidden_units": hidden_units,
    #     "state_dict": model.state_dict(),
    # }

    # torch.save(checkpoint, save_dir + f"{arch}_checkpoint.pth")

    # Save the mapping of classes to indices
    model.class_to_idx = class_to_idx
    # model.class_to_idx = image_datasets['train'].class_to_idx

    # Create a checkpoint dictionary
    checkpoint = {
        "class_to_idx": model.class_to_idx,
        "state_dict": model.state_dict(),
        "classifier": model.classifier,
        # 'optimizer_state': optimizer.state_dict(),
        # 'num_epochs': num_epochs
    }

    # Save the checkpoint
    torch.save(checkpoint, save_dir + "checkpoint.pth")


# def load_checkpoint(filepath):
#     checkpoint = torch.load(filepath)
#     arch = checkpoint['arch']
#     hidden_units = checkpoint['hidden_units']
#     model = build_model(arch, hidden_units)
#     model.load_state_dict(checkpoint['state_dict'])

#     return model


def load_checkpoint(filepath):
    # Load the saved file
    checkpoint = torch.load(filepath)

    # Download pretrained model
    model = models.vgg16(pretrained=True)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # Load stuff from checkpoint
    model.class_to_idx = checkpoint["class_to_idx"]
    model.classifier = checkpoint["classifier"]
    model.load_state_dict(checkpoint["state_dict"])

    return model


# def predict(image, model, topk=1):
#     model.eval()
#     with torch.no_grad():
#         output = model(image)
#         ps = torch.exp(output)
#         top_p, top_class = ps.topk(topk, dim=1)

#     return top_p[0].tolist(), top_class[0].tolist()


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

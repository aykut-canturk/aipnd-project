import argparse
import json
import numpy as np
import torch
from torch import nn
from torchvision import models
from PIL import Image

def load_checkpoint(filepath):
    model = models.vgg16(pretrained=True)
    model.classifier = nn.Sequential(nn.Linear(25088, 4096),
                                     nn.ReLU(),
                                     nn.Dropout(0.5),
                                     nn.Linear(4096, 1000),
                                     nn.ReLU(),
                                     nn.Dropout(0.5),
                                     nn.Linear(1000, 102),
                                     nn.LogSoftmax(dim=1))
    model.load_state_dict(torch.load(filepath))
    return model

def process_image(image):
    img = Image.open(image)
    img = img.resize((256, 256))
    img = img.crop((16, 16, 240, 240))
    img = np.array(img) / 255
    img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    img = img.transpose((2, 0, 1))
    return img

def predict(image_path, model, topk, device):
    image = process_image(image_path)
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image = image.unsqueeze(0)
    image = image.to(device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        output = model.forward(image)
    ps = torch.exp(output)
    top_p, top_class = ps.topk(topk, dim=1)
    return top_p, top_class

def main():
    parser = argparse.ArgumentParser(description='Predict flower name from an image.')
    parser.add_argument('input', type=str, help='Path to image')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint')
    parser.add_argument('--top_k', type=int, default=1, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Path to category to name mapping json')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")

    model = load_checkpoint(args.checkpoint)

    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    top_p, top_class = predict(args.input, model, args.top_k, device)

    for i in range(args.top_k):
        print(f"Prediction {i+1}:")
        print(f"Class: {cat_to_name[str(top_class[0][i].item())]}")
        print(f"Probability: {top_p[0][i].item()}")

if __name__ == "__main__":
    main()
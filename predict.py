import argparse
import json
import torch
from torchvision import models, transforms
from PIL import Image

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.classifier = checkpoint['classifier']
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("Unsupported architecture")
    return model, checkpoint['class_to_idx']

def process_image(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor()])
    image = transform(image)
    return image

def predict(image_path, model, topk, device):
    image = process_image(image_path)
    image = image.unsqueeze(0)
    image = image.to(device)

    model.eval()
    model.to(device)
    with torch.no_grad():
        output = model.forward(image)

    probs, indices = torch.topk(output, topk)
    probs = probs.exp()

    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    classes = [idx_to_class[i] for i in indices[0].tolist()]

    return probs[0].tolist(), classes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', default='flowers/test/1/image_06743.jpg')
    parser.add_argument('checkpoint', default='checkpoint.pth')
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--category_names', type=str)
    parser.add_argument('--gpu', action='store_true')

    args = parser.parse_args()

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    model, class_to_idx = load_checkpoint(args.checkpoint)
    model.class_to_idx = class_to_idx

    probs, classes = predict(args.input, model, args.top_k, device)
    print('Predicted class:', classes)
    print('Probabilities:', probs)
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        classes = [cat_to_name[i] for i in classes]

    print('Classes and probabilities:', list(zip(classes, probs)))

if __name__ == "__main__":
    main()

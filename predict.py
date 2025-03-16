import argparse
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import json
import os

def get_args():
    parser = argparse.ArgumentParser(description="Predict an image using a trained model")
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--top_k", type=int, default=5, help="Return top K most likely classes")
    parser.add_argument("--category_names", type=str, help="Path to JSON file mapping categories to names")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    return parser.parse_args()

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location="cuda" if torch.cuda.is_available() else "cpu")
    model = models.__dict__[checkpoint['arch']](pretrained=True)
    
    if checkpoint['arch'] == 'vgg16':
        model.classifier = torch.nn.Sequential(
            torch.nn.Linear(25088, checkpoint['hidden_units']),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(checkpoint['hidden_units'], 102),
            torch.nn.LogSoftmax(dim=1)
        )
    elif checkpoint['arch'] == 'resnet18':
        model.fc = torch.nn.Sequential(
            torch.nn.Linear(512, checkpoint['hidden_units']),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(checkpoint['hidden_units'], 102),
            torch.nn.LogSoftmax(dim=1)
        )

    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.eval()
    return model

def process_image(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def predict(args):
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    
    model = load_checkpoint(args.checkpoint).to(device)
    image = process_image(args.image_path).to(device)
    
    with torch.no_grad():
        output = model(image)
    probabilities = torch.exp(output)
    top_probs, top_classes = probabilities.topk(args.top_k, dim=1)
    
    top_probs = top_probs.squeeze().tolist()
    top_classes = top_classes.squeeze().tolist()
    
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        idx_to_class = {v: k for k, v in model.class_to_idx.items()}
        top_classes = [cat_to_name[idx_to_class[i]] for i in top_classes]
    
    print(f"Top {args.top_k} Predictions:")
    for prob, cls in zip(top_probs, top_classes):
        print(f"{cls}: {prob:.3f}")

if __name__ == "__main__":
    args = get_args()
    predict(args)

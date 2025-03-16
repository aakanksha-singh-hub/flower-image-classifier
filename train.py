import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os

def get_args():
    parser = argparse.ArgumentParser(description="Train an image classifier")
    parser.add_argument("data_dir", type=str, help="Path to dataset")
    parser.add_argument("--save_dir", type=str, default=".", help="Directory to save checkpoint")
    parser.add_argument("--arch", type=str, default="vgg16", choices=["vgg16", "resnet18"], help="Model architecture")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden_units", type=int, default=512, help="Number of hidden units")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    return parser.parse_args()

def train_model(args):
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    
    # Data preprocessing
    train_dir = os.path.join(args.data_dir, 'train')
    valid_dir = os.path.join(args.data_dir, 'valid')
    
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_data = ImageFolder(train_dir, transform=transform)
    valid_data = ImageFolder(valid_dir, transform=transform)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=32)
    
    # Model setup
    model = models.__dict__[args.arch](pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    
    if args.arch == "vgg16":
        model.classifier = nn.Sequential(
            nn.Linear(25088, args.hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(args.hidden_units, 102),
            nn.LogSoftmax(dim=1)
        )
    elif args.arch == "resnet18":
        model.fc = nn.Sequential(
            nn.Linear(512, args.hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(args.hidden_units, 102),
            nn.LogSoftmax(dim=1)
        )
    
    model.to(device)
    
    # Loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    
    # Training loop
    for epoch in range(args.epochs):
        train_loss = 0
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {train_loss/len(train_loader):.4f}")
    
    # Save checkpoint
    checkpoint = {
        'arch': args.arch,
        'state_dict': model.state_dict(),
        'hidden_units': args.hidden_units,
        'class_to_idx': train_data.class_to_idx
    }
    save_path = os.path.join(args.save_dir, 'checkpoint.pth')
    os.makedirs(args.save_dir, exist_ok=True)  # Ensure the directory exists
    torch.save(checkpoint, save_path)

    print("Model saved successfully!")

if __name__ == "__main__":
    args = get_args()
    train_model(args)

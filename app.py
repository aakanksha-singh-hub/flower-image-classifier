import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import json
import torchvision.models as models
import torch.nn as nn

def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    

    model = models.__dict__[checkpoint['arch']](pretrained=True)

    for param in model.parameters():
        param.requires_grad = False # freeze the parameters

    if checkpoint['arch'] == "vgg16":
        model.classifier = nn.Sequential(
            nn.Linear(25088, checkpoint['hidden_units']),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(checkpoint['hidden_units'], 102),
            nn.LogSoftmax(dim=1)
        )
    elif checkpoint['arch'] == "resnet18":
        model.fc = nn.Sequential(
            nn.Linear(512, checkpoint['hidden_units']),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(checkpoint['hidden_units'], 102),
            nn.LogSoftmax(dim=1)
        )

    model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    return model


# Process image
def process_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Predict function
def predict(image, model, top_k=5):
    with torch.no_grad():
        output = model(image)
        probabilities, indices = torch.exp(output).topk(top_k)
    return probabilities.squeeze().tolist(), indices.squeeze().tolist()

# Load model
model = load_model("checkpoint.pth")


with open("cat_to_name.json", "r") as f:
    cat_to_name = json.load(f)

st.title("Flower Image Classifier 🌸")
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    
    image_tensor = process_image(image)
    probs, classes = predict(image_tensor, model)

    st.subheader("Top Predictions:")
    for i, (prob, class_idx) in enumerate(zip(probs, classes)):
        flower_name = cat_to_name.get(str(class_idx), "Unknown")
        st.write(f"{i+1}. {flower_name}: {prob:.3f}")



# Flower Image Classifier 🌸

## Overview
This project is an **image classifier** built using **PyTorch** and **Streamlit** that identifies different types of flowers. It uses a **pretrained deep learning model (VGG16/ResNet)** to classify flower images uploaded by the user.

## Features
- Train a deep learning model using **transfer learning**.
- Save and load trained models using **checkpoints**.
- Predict flower species from an image.
- Deploy the model using **Streamlit**.
- Interactive **web app** for easy image upload and classification.

## Project Structure
```
aipnd-project-AkankshaSingh/
│-- flower_data/                 # Dataset
│-- .gitignore                    # Ignored files
│-- app.py                        # Streamlit web app
│-- cat_to_name.json              # Class to flower name mapping
│-- checkpoint.pth                # Saved model checkpoint
│-- Image_Classifier_Project.ipynb # Jupyter Notebook
│-- LICENSE                       # License info
│-- predict.py                    # Script to make predictions
│-- README.md                     # Project documentation
│-- requirements.txt               # Python dependencies
│-- train.py                       # Script to train the model
```

## Installation
### 1️⃣ Clone the Repository
```sh
git clone https://github.com/yourusername/flower-image-classifier.git
cd flower-image-classifier
```
### 2️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

## Training the Model
To train the model, run:
```sh
python train.py --data_dir flower_data --epochs 5 --gpu
```

## Making Predictions
Use `predict.py` to classify an image:
```sh
python predict.py --image_path path/to/image.jpg --checkpoint checkpoint.pth
```

## Running the Web App 🌐
To start the Streamlit app:
```sh
streamlit run app.py
```

## Deployment on Streamlit Cloud
1. **Push your code to GitHub**.
2. **Create a Streamlit Cloud account**.
3. **Deploy the repository**.
4. Ensure `requirements.txt` includes `torch`, `torchvision`, `streamlit`, and other necessary libraries.

## Example Usage
1. **Upload an image**.
2. **The model predicts the flower species**.
3. **Displays top predicted classes with confidence scores**.

## Requirements
- Python 3.x
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- Pillow
- Streamlit

## License
This project is licensed under the **MIT License**.

---
Made with ❤️ by Aakanksha Singh



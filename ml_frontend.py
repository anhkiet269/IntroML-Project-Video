import streamlit as st
import gdown
# import the necessary packages for image recognition
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import pandas as pd
import urllib

#Transform for image preprocessing
valid_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    transforms.Resize((224,224),antialias=True)]
)

#Predicts Image
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
def predict(model, img, plant, device):
    img = img.to(device)
    predictions = model(img)
    if plant=="Apple":
        columns=['healthy','multiple_diseases','rust','scab']
    if plant=="Cassava":
        columns=['Bacterial Blight','Brown Streak Disease','Green Mottle','Mosaic Disease','Healthy']
    score = nn.Softmax(1)(predictions).detach().cpu().numpy()
    output = pd.DataFrame(100*score, columns=columns)
    result =pd.DataFrame()
    result = pd.concat([result, output], ignore_index=True)
    return result

# set page layout
st.set_page_config(
    page_title="Plant Diseases Classification App",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Plant Diseases Classification")
st.write("\nOur website does classification of leaf diseases of two types of plants are apple and cassava")
st.sidebar.subheader("Choosing a type of plants")
models_list = ["Apple", "Cassava"]
network = st.sidebar.selectbox("Apple or Cassava", models_list)

# component to upload images
uploaded_file = st.sidebar.file_uploader(
    "Choose an image to classify", type=["jpg", "jpeg", "png"]
)
# component for toggling code
show_code = st.sidebar.checkbox("Show Code")

if uploaded_file:
    #Load pretrained-models
    num_classes=0
    def get_model(num_classes):
        model = models.resnet50(pretrained= True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features=in_features, out_features=num_classes)
        return model
    #Apple model
    if network=="Apple":
        num_classes=4
    if network=="Cassava":
        num_classes=5
    model = get_model(num_classes = num_classes)
    model.eval()
    model.to(device)
    if network=="Apple":
        if torch.cuda.is_available():
            model.load_state_dict(torch.load('./apple_leaf_classifier_resnet50.pth'))
        else:
            model.load_state_dict(torch.load('./apple_leaf_classifier_resnet50.pth', map_location=torch.device('cpu')))
    if network=="Cassava":
        if torch.cuda.is_available():
            model.load_state_dict(torch.load('./cassava_leaf_classifier_resnet50.pth'))
        else:
            model.load_state_dict(torch.load('./cassava_leaf_classifier_resnet50.pth', map_location=torch.device('cpu')))

    # load the input image using PIL image
    image = Image.open(uploaded_file)
    image = image.convert("RGB")
    image = valid_transforms(image)
    image = image.unsqueeze(0)

    preds = predict(model,image,network,device)

    st.image(uploaded_file)
    st.subheader(f"Predictions for {network} image")
    st.dataframe(preds)

#Download a single file and make its content available as a string.
@st.cache(show_spinner=False)
def get_file_content_as_string():
    url = "https://raw.githubusercontent.com/VTaPo/intro2ML/main/ml_frontend.py"
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")


if show_code:
    st.code(get_file_content_as_string())

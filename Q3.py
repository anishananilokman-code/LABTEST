import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import requests
import pandas as pd

# ------------------------------- #
# Page Setup
# ------------------------------- #
st.set_page_config(page_title="Real-Time Image Classifier", layout="wide")

st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>📷 Live Image Classification</h1>",
    unsafe_allow_html=True
)
st.caption("Using Pretrained ResNet-18 Model | Webcam Image Classification (ImageNet)")

# ------------------------------- #
# Step 1: Library setup + device
# ------------------------------- #
device = torch.device("cpu")

# ------------------------------- #
# Step 2: Download ImageNet class labels
# ------------------------------- #
IMAGENET_CLASSES_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
response = requests.get(IMAGENET_CLASSES_URL)
class_labels = response.text.splitlines()

# ------------------------------- #
# Step 3: Load pretrained ResNet-18 model
# ------------------------------- #
cnn_model = models.resnet18(pretrained=True)
cnn_model.eval()
cnn_model.to(device)

# ------------------------------- #
# Step 4: Image preprocessing pipeline
# ------------------------------- #
transform_pipeline = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ------------------------------- #
# Step 5: Capture image from webcam
# ------------------------------- #
st.subheader("📸 Capture Image from Webcam")
captured_img = st.camera_input("Take a picture")

if captured_img is not None:
    img = Image.open(captured_img).convert("RGB")

    # Display captured image
    st.image(img, caption="Captured Image", use_container_width=True)

    # Preprocess image into batch tensor
    tensor_input = transform_pipeline(img)
    tensor_input_batch = tensor_input.unsqueeze(0).to(device)  # [1, 3, 224, 224]

    # ------------------------------- #
    # Step 6: Model prediction + softmax
    # ------------------------------- #
    with torch.no_grad():
        model_output = cnn_model(tensor_input_batch)

    softmax_probs = F.softmax(model_output[0], dim=0)

    # Top-5 Predictions
    top5_probs, top5_indices = torch.topk(softmax_probs, 5)

    # Prepare results table
    prediction_data = pd.DataFrame({
        "Class Label": [class_labels[idx] for idx in top5_indices],
        "Probability": top5_probs.cpu().numpy()
    })

    # Display results
    st.subheader("🔍 Top-5 Predictions")
    st.dataframe(prediction_data, use_container_width=True)

    st.bar_chart(prediction_data.set_index("Class Label"), horizontal=True)

    top_prediction = prediction_data.iloc[0]
    st.success(
        f"**Top Prediction:** {top_prediction['Class Label']} ({top_prediction['Probability']:.2%})"
    )

    st.info("Softmax probabilities represent the confidence level of each predicted class.")
else:
    st.info("Please capture an image using the webcam to see predictions.")

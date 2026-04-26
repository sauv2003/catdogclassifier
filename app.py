import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from model import Cat_Dog_Classifier # Import your class

# 1. Load the model
@st.cache_resource # Keeps the model in memory so it doesn't reload every click
def load_model():
    model = Cat_Dog_Classifier()
    model.load_state_dict(torch.load("cat_dog_classifier.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# 2. Setup Image Preprocessing (Must match your training transforms)
transform = transforms.Compose([
    transforms.Resize((224, 224)), # Or whatever size you used
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. Streamlit UI
st.title("🐱 Cat vs Dog 🐶 Classifier")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess and Predict
    img_tensor = transform(image).unsqueeze(0) 
    with torch.no_grad():
        output = model(img_tensor)
        prediction = torch.argmax(output, dim=1).item()
        
    classes = ['Cat', 'Dog']
    st.write(f"## Prediction: {classes[prediction]}")
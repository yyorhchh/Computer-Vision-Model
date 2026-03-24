import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io
import cv2
import numpy as np

def crop_and_resize_memory(img_bytes, target_size):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        img = img[y:y+h, x:x+w]
        
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    
    # 3. Apply CLAHE normalization for Medical Imaging
    gray_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_resized)
    
    # 4. Convert back to RGB for ToTensor compatibility
    img_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    
    return Image.fromarray(img_rgb)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Reconstructed CustomCNN Architecture
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize Models
custom_model = CustomCNN()
resnet_model = models.resnet50(weights="IMAGENET1K_V1")
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 2)

# Load State Dicts
CUSTOM_PATH = r"C:\Computer vision\model\best_custom_cnn.pth"
RESNET_PATH = r"C:\Computer vision\model\best_resnet50.pth"

try:
    custom_model.load_state_dict(torch.load(CUSTOM_PATH, map_location='cpu'))
    resnet_model.load_state_dict(torch.load(RESNET_PATH, map_location='cpu'))
    print("Models loaded successfully in FastAPI.")
except Exception as e:
    print(f"Error loading models in FastAPI: {e}")

custom_model.eval()
resnet_model.eval()

# Image Preprocessing (Standard PIL based) matching User's unified Training Pipeline
transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

classes = ["not_pneumonia", "pneumonia"]

@app.post("/predict")
async def predict(image: UploadFile = File(...), model: str = Form("custom")):
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    
    try:
        processed_img = crop_and_resize_memory(img_bytes, target_size=(224, 224))
        img = processed_img if processed_img else img
    except Exception as e:
        print(f"Crop warning: {e}")
    
    if model == 'resnet':
        input_tensor = transform_pipeline(img).unsqueeze(0)
        net = resnet_model
    else:
        input_tensor = transform_pipeline(img).unsqueeze(0)
        net = custom_model
        
    with torch.no_grad():
        outputs = net(input_tensor)
        print(f"[{model}] Raw logits: {outputs.tolist()}")
        probabilities = torch.softmax(outputs / 3.0, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    conf_val = float(confidence.item())
    class_name = classes[predicted.item()]
    print(f"[{model}] Predicted Index (0 or 1): {predicted.item()} --> Confidence: {conf_val:.4f}")
        
    return {
        'class': class_name,
        'confidence': conf_val,
        'all_probs': probabilities[0].tolist()
    }

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

# Serve static files (css, js, images)
app.mount("/", StaticFiles(directory="."), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

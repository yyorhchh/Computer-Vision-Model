import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import io
import cv2
import numpy as np
import base64
import torch.nn.functional as F

# --- Grad-CAM Utils ---
class SimpleGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.hook_f = target_layer.register_forward_hook(self.save_activation)
        self.hook_b = target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[0, class_idx]
        score.backward(retain_graph=True)
        
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        self.hook_f.remove()
        self.hook_b.remove()
        
        return cam.detach().cpu().numpy(), output

def apply_colormap_on_image(org_im, activation, colormap=cv2.COLORMAP_JET):
    activation = cv2.resize(activation, (org_im.shape[1], org_im.shape[0]))
    heatmap = np.uint8(255 * activation)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    superimposed_img = np.float32(heatmap) * 0.4 + np.float32(org_im) * 0.6
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    return superimposed_img

def image_to_base64(img_array):
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', img_bgr)
    return base64.b64encode(buffer).decode('utf-8')
# ----------------------

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def send_static(path):
    return send_from_directory('.', path)

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
        # Based on state_dict size [512, 2048], in_features = 2048
        # 2048 / 128 = 16, which means spatial dim is 4x4
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
    print("Models loaded successfully!")
except Exception as e:
    print(f"CRITICAL ERROR LOADING MODELS: {e}")
    raise e

custom_model.eval()
resnet_model.eval()

# Image Preprocessing (Standard PIL based) matching User's unified Training Pipeline
transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

classes = ["not_pneumonia", "pneumonia"]

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    model_type = request.form.get('model', 'custom')
    
    img_bytes = file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    
    # Try user's cropping function
    try:
        processed_img = crop_and_resize_memory(img_bytes, target_size=(224, 224))
        image = processed_img if processed_img else image
    except Exception as e:
        print(f"Crop warning: {e}")
    
    if model_type == 'resnet':
        input_tensor = transform_pipeline(image).unsqueeze(0)
        model = resnet_model
        target_layer = resnet_model.layer4[-1]
    else:
        input_tensor = transform_pipeline(image).unsqueeze(0)
        model = custom_model
        target_layer = custom_model.conv3
        
    with torch.set_grad_enabled(True):
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs / 3.0, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        # Fire up Grad-CAM for the predicted class
        cam_extractor = SimpleGradCAM(model, target_layer)
        grayscale_cam, _ = cam_extractor.generate(input_tensor, predicted.item())
        
    # Generate visual outputs
    org_img_np = np.array(image.resize((224, 224)))
    heatmap_img = apply_colormap_on_image(org_img_np, grayscale_cam)
    
    before_b64 = image_to_base64(org_img_np)
    after_b64 = image_to_base64(heatmap_img)
        
    conf_val = float(confidence.item())
    class_name = classes[predicted.item()]
    
    # Debug logits to console
    print(f"[{model_type}] Raw logits: {outputs.tolist()}")
    print(f"[{model_type}] Predicted Index (0 or 1): {predicted.item()} --> Confidence: {conf_val:.4f}")
        
    result = {
        'class': class_name,
        'confidence': conf_val,
        'all_probs': probabilities[0].tolist(),
        'before_img': "data:image/jpeg;base64," + before_b64,
        'after_img': "data:image/jpeg;base64," + after_b64
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(port=5000, debug=True)

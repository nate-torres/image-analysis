import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

# ──────────────────────────────────────────────────────────────────────────────
# 1) Model definition (unchanged from training)
class MultiLabelResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.base_model.fc = nn.Sequential(
            nn.Linear(self.base_model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.base_model(x)

# Label names
label_names = ["golden_hour","blue_hour","night"]

# ──────────────────────────────────────────────────────────────────────────────
# 3) Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = MultiLabelResNet(num_classes=len(label_names)).to(device)
model.load_state_dict(torch.load("lighting/time/lighting_time_model.pth", map_location=device))
model.eval()

# ──────────────────────────────────────────────────────────────────────────────
# 4) Inference transform (must match training!)
val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

def predict_light_time(image_path: str,
                  threshold: float = 0.5
                  ):
    # a) Network inference
    img = Image.open(image_path).convert("RGB")
    x   = val_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = model(x).squeeze().cpu().numpy()

    # b) Pick all labels whose sigmoid > threshold
    net_labels = [lbl for lbl, p in zip(label_names, probs) if p > threshold]

    # c) Debug: show raw sigmoid scores
    print("\nRaw sigmoid scores:")
    for lbl, p in zip(label_names, probs):
        print(f"  {lbl:15s}: {p:.3f}")

    # e) Assemble final labels
    final = net_labels.copy()

    return final

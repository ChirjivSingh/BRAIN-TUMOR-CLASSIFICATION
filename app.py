import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import streamlit as st

# Configure page
st.set_page_config(
    page_title="Brain Tumour Classifier",
    page_icon="🧠",
    layout="centered"
)

CLASS_MAP = {
    "1.0": "Meningioma (Class 1)",
    "2.0": "Glioma (Class 2)",
    "3.0": "Pituitary (Class 3)"
}

MODEL_CANDIDATE_PATHS = {
    "Custom CNN": ["custom_cnn_best.pth", "custom_cnn.pth"],
    "ResNet18": ["resnet18_best.pth", "resnet.pth", "resnet18.pth"],
    "EfficientNet-B0": ["efficientnet_b0_best.pth", "efficientnet.pth", "efficientnet_b0.pth"]
}


class CustomCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def resolve_weight_path(candidates):
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


@st.cache_resource
def load_all_models():
    models_dict = {}
    load_errors = {}

    # 1. Custom CNN
    cnn_path = resolve_weight_path(MODEL_CANDIDATE_PATHS["Custom CNN"])
    if cnn_path:
        try:
            cnn = CustomCNN(num_classes=3)
            checkpoint = torch.load(cnn_path, map_location="cpu")
            state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
            cnn.load_state_dict(state_dict)
            cnn.eval()
            models_dict["Custom CNN"] = cnn
        except Exception as e:
            load_errors["Custom CNN"] = str(e)
    else:
        load_errors["Custom CNN"] = "Weight file not found"

    # 2. ResNet18
    resnet_path = resolve_weight_path(MODEL_CANDIDATE_PATHS["ResNet18"])
    if resnet_path:
        try:
            resnet = models.resnet18()
            resnet.fc = nn.Linear(resnet.fc.in_features, 3)
            checkpoint = torch.load(resnet_path, map_location="cpu")
            state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
            resnet.load_state_dict(state_dict)
            resnet.eval()
            models_dict["ResNet18"] = resnet
        except Exception as e:
            load_errors["ResNet18"] = str(e)
    else:
        load_errors["ResNet18"] = "Weight file not found"

    # 3. EfficientNet-B0
    effnet_path = resolve_weight_path(MODEL_CANDIDATE_PATHS["EfficientNet-B0"])
    if effnet_path:
        try:
            effnet = models.efficientnet_b0()
            effnet.classifier[1] = nn.Linear(effnet.classifier[1].in_features, 3)
            checkpoint = torch.load(effnet_path, map_location="cpu")
            state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
            effnet.load_state_dict(state_dict)
            effnet.eval()
            models_dict["EfficientNet-B0"] = effnet
        except Exception as e:
            load_errors["EfficientNet-B0"] = str(e)
    else:
        load_errors["EfficientNet-B0"] = "Weight file not found"

    return models_dict, load_errors


def get_preprocessing():
    # Load exact metadata if available
    img_size = 224
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if os.path.exists("model_metadata.json"):
        try:
            with open("model_metadata.json", "r") as f:
                metadata = json.load(f)
                img_size = metadata.get("image_size", img_size)
                mean = metadata.get("normalization_mean", mean)
                std = metadata.get("normalization_std", std)
        except Exception:
            pass

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return transform


def predict(image, model):
    transform = get_preprocessing()
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        conf, pred_idx = torch.max(probabilities, dim=0)

    class_keys = ["1.0", "2.0", "3.0"]
    pred_key = class_keys[pred_idx.item()]
    class_label = CLASS_MAP.get(pred_key, f"Class {pred_key}")
    confidence_pct = conf.item() * 100.0

    return class_label, confidence_pct, probabilities.numpy()


def main():
    st.title("🧠 Brain Tumour Classification")
    st.write("Upload a brain MRI scan to predict the tumour type across all three trained models.")

    models_dict, load_errors = load_all_models()

    if load_errors:
        for m_name, err in load_errors.items():
            st.warning(f"Could not load {m_name}: {err}")

    uploaded_file = st.file_uploader(
        "Choose a brain MRI image...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded MRI Scan", use_container_width=True)
        except Exception as e:
            st.error(f"Invalid image file: {e}")
            return

        st.subheader("Model Predictions")
        cols = st.columns(3)

        model_order = ["Custom CNN", "ResNet18", "EfficientNet-B0"]

        for idx, model_name in enumerate(model_order):
            with cols[idx]:
                st.markdown(f"### {model_name}")
                if model_name in models_dict:
                    try:
                        pred_class, confidence, _ = predict(image, models_dict[model_name])
                        st.metric(label="Predicted Class", value=pred_class)
                        st.metric(label="Confidence", value=f"{confidence:.2f}%")
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
                else:
                    st.error("Model unavailable")


if __name__ == "__main__":
    main()

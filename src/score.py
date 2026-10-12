"""
Scoring script for the Azure ML managed online endpoint.

Azure ML calls init() once when the container starts and run() for every request.

Request body (JSON), either:
    {"image": "<base64-encoded jpg/png>"}
or  {"images": ["<base64>", "<base64>", ...]}

Response (JSON list, one item per image):
    [{"prediction": "Covid", "confidence": 0.97,
      "probabilities": {"Covid": 0.97, "Normal": 0.02, "Viral Pneumonia": 0.01}}]
"""
import base64
import glob
import io
import json
import logging
import os

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

model = None
classes = None
preprocess = None


def build_model(num_classes: int) -> nn.Module:
    """Must match the architecture used in 01_train_model.ipynb."""
    net = models.resnet50(weights=None)
    net.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(net.fc.in_features, num_classes))
    return net


def init():
    global model, classes, preprocess
    model_root = os.getenv("AZUREML_MODEL_DIR", ".")
    # The registered folder can be nested (e.g. model/covid_resnet50.pt), so search for it.
    meta_path = glob.glob(os.path.join(model_root, "**", "labels.json"), recursive=True)[0]
    weights_path = glob.glob(os.path.join(model_root, "**", "*.pt"), recursive=True)[0]

    with open(meta_path) as f:
        meta = json.load(f)
    classes = meta["classes"]

    model = build_model(len(classes))
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((meta["img_size"], meta["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(meta["mean"], meta["std"]),
    ])
    logging.info("Model loaded: %s | classes=%s", weights_path, classes)


def _decode(b64_string: str) -> Image.Image:
    if "," in b64_string[:100]:  # strip "data:image/png;base64," prefix if present
        b64_string = b64_string.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64_string))).convert("RGB")


def run(raw_data):
    try:
        data = json.loads(raw_data)
        images = data.get("images") or ([data["image"]] if "image" in data else None)
        if not images:
            return {"error": 'Send JSON like {"image": "<base64>"} or {"images": ["<base64>", ...]}'}
        batch = torch.stack([preprocess(_decode(s)) for s in images])
        with torch.no_grad():
            probs = torch.softmax(model(batch), dim=1)

        results = []
        for p in probs:
            idx = int(p.argmax())
            results.append({
                "prediction": classes[idx],
                "confidence": round(float(p[idx]), 4),
                "probabilities": {c: round(float(v), 4) for c, v in zip(classes, p)},
            })
        return results
    except Exception as e:  # return the error so it shows up in the response while debugging
        logging.exception("Scoring failed")
        return {"error": str(e)}

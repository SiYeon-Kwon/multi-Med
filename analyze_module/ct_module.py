import pydicom
import os
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image

class CTAnalyzer:
    def __init__(self, model_path=None):
        self.model = resnet18(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 2)
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def load_series(self, dicom_folder):
        slices = [pydicom.dcmread(os.path.join(dicom_folder, f)) for f in sorted(os.listdir(dicom_folder)) if f.endswith(".dcm")]
        slices = sorted(slices, key=lambda s: float(s.ImagePositionPatient[2]))
        images = [Image.fromarray(s.pixel_array).convert("RGB") for s in slices]
        return images

    def analyze(self, dicom_folder):
        images = self.load_series(dicom_folder)
        predictions = []
        for img in images:
            tensor = self.transform(img).unsqueeze(0)
            with torch.no_grad():
                output = self.model(tensor)
                pred = torch.argmax(output, dim=1).item()
                predictions.append(pred)
        majority = max(set(predictions), key=predictions.count)
        return majority, images[0], self.model

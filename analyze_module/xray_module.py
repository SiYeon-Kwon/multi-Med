import torch
import torchvision.transforms as transforms
from torchvision.models import swin_t
from PIL import Image

class XRayAnalyzer:
    def __init__(self, model_path=None):
        self.model = swin_t(weights='IMAGENET1K_V1')
        self.model.head = torch.nn.Linear(self.model.head.in_features, 2)  # 예: 2-class 분류
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def analyze(self, image_path):
        image = Image.open(image_path).convert('RGB')
        tensor = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            output = self.model(tensor)
        prediction = torch.argmax(output, dim=1).item()
        return prediction, tensor, self.model
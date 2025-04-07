import torch
import torch.nn as nn

class ECGNet(nn.Module):
    def __init__(self, input_size=5000, num_classes=5):
        super(ECGNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=7, padding=3)
        self.fc1 = nn.Linear(64 * (input_size // 4), 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def analyze_ecg(ecg_series):
    model = ECGNet()
    model.eval()
    with torch.no_grad():
        ecg_tensor = torch.tensor(ecg_series).unsqueeze(0).unsqueeze(0).float()
        output = model(ecg_tensor)
    return torch.softmax(output, dim=1).squeeze().tolist()
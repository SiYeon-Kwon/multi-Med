class EEGNet(nn.Module):
    def __init__(self, input_size=5000, num_classes=4):
        super(EEGNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(32 * (input_size // 4), 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def analyze_eeg(eeg_series):
    model = EEGNet()
    model.eval()
    with torch.no_grad():
        eeg_tensor = torch.tensor(eeg_series).unsqueeze(0).unsqueeze(0).float()
        output = model(eeg_tensor)
    return torch.softmax(output, dim=1).squeeze().tolist()
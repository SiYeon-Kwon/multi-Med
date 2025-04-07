'''import torch
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output)

        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward()

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.detach()[0]

        for i in range(activations.shape[0]):
            activations[i, ...] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        return heatmap'''
    
import matplotlib.pyplot as plt
import torch
import numpy as np
    
def generate_gradcam(model, data, data_type):
    importance = abs(torch.tensor(data))
    plt.plot(importance.numpy() if importance.ndim == 1 else importance.squeeze().numpy())
    plt.title(f"{data_type.upper()} Grad-CAM")
    filename = f"gradcam_{data_type}.png"
    plt.savefig(filename)
    return filename

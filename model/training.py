import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os

class CifarFPGA(nn.Module):
    def __init__(self):
        super(CifarFPGA, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64 * 4 * 4, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        return self.fc(x)

def export_assets(model, folder="cifar_weights"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    layers = {"layer1": model.conv1, "layer2": model.conv2, "layer3": model.conv3}
    header_path = os.path.join(folder, "weights.h")
    
    with open(header_path, "w") as f_h:
        f_h.write("#ifndef WEIGHTS_H\n#define WEIGHTS_H\n\n")
        
        for name, layer in layers.items():
            w = layer.weight.data.numpy().astype(np.float32)
            b = layer.bias.data.numpy().astype(np.float32)
            
            f_h.write(f"static const float {name}_w[] = {{{', '.join(map(str, w.flatten()))}}};\n")
            f_h.write(f"static const float {name}_b[] = {{{', '.join(map(str, b.flatten()))}}};\n\n")
        
        fc_w = model.fc.weight.data.numpy().astype(np.float32)
        fc_b = model.fc.bias.data.numpy().astype(np.float32)
        
        f_h.write(f"static const float fc_w[] = {{{', '.join(map(str, fc_w.flatten()))}}};\n")
        f_h.write(f"static const float fc_b[] = {{{', '.join(map(str, fc_b.flatten()))}}};\n")
        f_h.write("\n#endif")
    
    print(f"Exported weights.h to {folder}/")

transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

model = CifarFPGA().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

epochs = 10
for epoch in range(epochs):
    running_loss, correct, total = 0.0, 0, 0
    
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(trainloader):.4f} Accuracy: {100*correct/total:.2f}%")

model = model.to("cpu")
export_assets(model)

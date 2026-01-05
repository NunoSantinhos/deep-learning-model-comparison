import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import utils

data = utils.load_data()

train_img = torch.tensor(data['train_X']).float().permute(0, 3, 1, 2)
train_masks = torch.tensor(data['train_masks']).float()

test_img = torch.tensor(data['test_X']).float().permute(0, 3, 1, 2)
test_masks = torch.tensor(data['test_masks']).float()

RANDOM_SEED = 1
NUM_EPOCHS = 100
BATCH_SIZE = 32
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_dataset = TensorDataset(train_img, train_masks)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class U_Net(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(U_Net, self).__init__()
        self.enc1 = ConvBlock(input_channels, 16)
        self.enc2 = ConvBlock(16, 32)
        self.enc3 = ConvBlock(32, 64)
        self.enc4 = ConvBlock(64, 128)
        self.enc5 = ConvBlock(128, 256)

        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(32, 16)

        self.final = nn.Conv2d(16, output_channels, kernel_size=1)

    def forward(self, x):

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(e5), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.final(d1)
    

    def predict(self, x, threshold=0.5, device='cpu'):
        
        self.eval()
        
        with torch.no_grad():
            logits = self(x)
            probs = torch.sigmoid(logits)
            predictions = probs > threshold
        return predictions

  



'''
torch.manual_seed(RANDOM_SEED)
model = U_Net(input_channels=3, output_channels=1)
model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.BCEWithLogitsLoss()

accuracy_per_epoch = []

for epoch in range(NUM_EPOCHS):
    start_time = time.time()
    epoch_loss = 0
    correct_predictions = 0
    total_predictions = 0

    model.train()
    for inputs, target_masks in train_loader:
        inputs, target_masks = inputs.to(DEVICE), target_masks.to(DEVICE).permute(0, 3, 1, 2)  # [batch_size, channels, height, width]

        outputs = model(inputs)
        loss = loss_fn(outputs, target_masks)
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predictions = torch.sigmoid(outputs) > 0.5
        correct_predictions += (predictions == target_masks).sum().item()
        total_predictions += target_masks.numel()

    accuracy = correct_predictions / total_predictions * 100
    accuracy_per_epoch.append(accuracy)

    epoch_duration = time.time() - start_time
    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {epoch_loss / len(train_loader):.4f}, Accuracy: {accuracy:.2f}%, Duration: {epoch_duration:.2f} seconds")

model.eval()

with torch.no_grad():
    test_inputs = test_img.to(DEVICE)
    test_targets = test_masks.to(DEVICE)

    test_outputs = model(test_inputs)

    test_predictions = torch.sigmoid(test_outputs) > 0.5
    test_correct_predictions = (test_predictions.permute(0, 2, 3, 1) == test_targets).sum().item()
    test_accuracy = (test_correct_predictions /  test_targets.numel()) * 100

print(test_correct_predictions, test_targets.numel())
print(f"Test Accuracy: {test_accuracy:.2f}%")

torch.save(model.state_dict(), 'modelo_segmentacao.pth')
print("Modelo completo salvo como 'modelo_segmentacao.pth'")

plt.figure(figsize=(10, 5))
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(test_img[i].permute(1, 2, 0).cpu().numpy())
    plt.title("Input Image")

    plt.subplot(2, 5, i + 6)
    plt.imshow(test_predictions[i][0].cpu().numpy(), cmap='gray')
    plt.title("Predicted Mask")

plt.tight_layout()
plt.show()
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import mobilenet_v2
import utils

# Carregar os dados
data = utils.load_data()

train_img = torch.tensor(data['train_X']).float().permute(0, 3, 1, 2)
train_labels = torch.tensor(data['train_labels']).float()

test_img = torch.tensor(data['test_X']).float().permute(0, 3, 1, 2)
test_labels = torch.tensor(data['test_labels']).float()

# Configurações
RANDOM_SEED = 1
NUM_EPOCHS = 500
BATCH_SIZE = 256
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Dataset e DataLoader
train_dataset = TensorDataset(train_img, train_labels)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataset = TensorDataset(test_img, test_labels)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Modelo original
class MultiRotulo(nn.Module):
    def __init__(self, input_shape=(64, 64, 3), num_classes=10):
        super(MultiRotulo, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_shape[2], out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(128 * 8 * 8, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = x.reshape(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

# Treinamento do modelo original
model_original = MultiRotulo(input_shape=(64, 64, 3), num_classes=10)
model_original = model_original.to(DEVICE)
optimizer = torch.optim.Adam(model_original.parameters(), lr=0.001)
loss_fn = torch.nn.BCEWithLogitsLoss()

for epoch in range(NUM_EPOCHS):
    model_original.train()
    for inputs, target_classes in train_loader:
        inputs, target_classes = inputs.to(DEVICE), target_classes.to(DEVICE)

        outputs = model_original(inputs)
        loss = loss_fn(outputs, target_classes)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Avaliação no conjunto de teste do modelo original
model_original.eval()
correct_predictions_original = 0
total_predictions_original = 0
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        outputs = model_original(inputs)
        predictions = torch.sigmoid(outputs) > 0.5
        correct_predictions_original += (predictions == targets).sum().item()
        total_predictions_original += targets.numel()

test_accuracy_original = correct_predictions_original / total_predictions_original * 100

torch.save(model_original.state_dict(), 'modelo_multirotulo.pth')
print("Modelo completo salvo como 'modelo_multirotulo.pth'")

# Avaliação no conjunto de teste do MobileNetV2 pré-treinado
mobilenet = mobilenet_v2(pretrained=True)
mobilenet.classifier[1] = nn.Linear(mobilenet.last_channel, train_labels.shape[1])
mobilenet = mobilenet.to(DEVICE)
mobilenet.eval()

correct_predictions_mobilenet = 0
total_predictions_mobilenet = 0
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        outputs = mobilenet(inputs)
        predictions = torch.sigmoid(outputs) > 0.5
        correct_predictions_mobilenet += (predictions == targets).sum().item()
        total_predictions_mobilenet += targets.numel()

test_accuracy_mobilenet = correct_predictions_mobilenet / total_predictions_mobilenet * 100

# Gráfico comparativo
models = ['Multirotulo', 'MobileNetV2']
accuracies = [test_accuracy_original, test_accuracy_mobilenet]

plt.figure(figsize=(8, 6))
plt.bar(models, accuracies, color=['blue', 'green'])
plt.xlabel('Modelos')
plt.ylabel('Accuracy no Conjunto de Teste (%)')
plt.title('Comparação de Desempenho no Conjunto de Teste')
plt.ylim(0, 100)
plt.savefig('multirotulo_vs_mobilenetv2.png')
plt.show()

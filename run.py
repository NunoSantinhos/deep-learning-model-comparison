import utils
import numpy as np
import torch
from segmentacao import U_Net

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


''' For multi label classification, use BCELoss or BCEWithLogitsLoss.'''

batch_size = 2
num_classes = 11
loss_fn = torch.nn.BCELoss()

outputs_before_sigmoid = torch.randn(batch_size, num_classes)
sigmoid_outputs = torch.sigmoid(outputs_before_sigmoid)
target_classes = torch.randint(0, 2, (batch_size, num_classes))  # randints in [0, 2).

loss = loss_fn(sigmoid_outputs, target_classes.float())

# alternatively, use BCE with logits, on outputs before sigmoid.
loss_fn_2 = torch.nn.BCEWithLogitsLoss()
loss2 = loss_fn_2(outputs_before_sigmoid, target_classes.float())
assert round(loss.item(), 3) == round(loss2.item(), 3)

data = utils.load_data()

train_img = data['train_X'][:]
test_img = data['test_X'][:]
train_masks = data['train_masks'][:]
test_masks = data['test_masks'][:]
train_class = data['train_classes'][:]
test_class = data['test_classes'][:]
train_labels = data['train_labels'][:]
test_labels = data['test_labels'][:]


#print(train_masks[0:10])

img = (data['test_X'][:20] * 255).astype(np.uint8)
utils.images_to_pic('test_set.png', img, width=10)

predicts_x = (data['test_X'])
#compare_masks('test_compare.png',data['test_masks'],predicts)
#utils.overlay_masks('test_overlay.png', np.expand_dims(test_img[0], axis=0), np.expand_dims(test_class[0], axis=0), width=1)

model = U_Net(input_channels=3, output_channels=1)

model.load_state_dict(torch.load('modelo_segmentacao.pth'))


# Envie o modelo para o dispositivo
#model = model.to(DEVICE)

model.eval()


# Geração das previsões
test_inputs = torch.tensor(data['test_X']).float().permute(0, 3, 1, 2).to(DEVICE)
predicts = model.predict(test_inputs).cpu().numpy()  # Converta para numpy array

# Ajuste de dimensões para corresponder a 'test_X'
predicts = np.transpose(predicts, (0, 2, 3, 1))  # [batch_size, height, width, channels]

# Remover dimensões extras, se existirem
if predicts.shape[-1] != 1:
    predicts = np.expand_dims(predicts[..., 0], axis=-1)  # Garantir que o último eixo é 1


utils.overlay_masks('test_overlay.png', data['test_X'][:20], predicts[:20], width=10)
utils.compare_masks('test_compare.png',data['test_masks'][:20],predicts[:20], width=10)

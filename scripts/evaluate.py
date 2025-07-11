import torch
from models.battery_gpt import BatteryGPT
from data.load_data import load_test_data
import yaml
from sklearn.metrics import mean_absolute_error

with open('../configs/finetune_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

model = BatteryGPT(**config['model_params']).cuda()
model.load_state_dict(torch.load('../models/battery_gpt_finetuned.pth'))
model.eval()

data_loader = load_test_data(config['data']['test_path'])

soh_preds, rul_preds = [], []
soh_labels, rul_labels = [], []

with torch.no_grad():
    for inputs, soh_label, rul_label in data_loader:
        inputs = [inp.cuda() for inp in inputs]
        soh_pred, rul_pred = model(inputs)
        soh_preds.extend(soh_pred.cpu().numpy())
        rul_preds.extend(rul_pred.cpu().numpy())
        soh_labels.extend(soh_label.numpy())
        rul_labels.extend(rul_label.numpy())

soh_mae = mean_absolute_error(soh_labels, soh_preds)
rul_mae = mean_absolute_error(rul_labels, rul_preds)

print(f'SOH MAE: {soh_mae}, RUL MAE: {rul_mae}')

import torch
from models.battery_gpt import BatteryGPT
from data.load_data import load_finetune_data
import yaml

with open('../configs/finetune_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

model = BatteryGPT(**config['model_params']).cuda()
model.load_state_dict(torch.load('../models/battery_gpt_pretrained.pth'))

optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'])
criterion = nn.MSELoss()

data_loader = load_finetune_data(config['data']['path'], config['train']['batch_size'])

for epoch in range(config['train']['epochs']):
    model.train()
    for inputs, soh_labels, rul_labels in data_loader:
        inputs = [inp.cuda() for inp in inputs]
        soh_labels, rul_labels = soh_labels.cuda(), rul_labels.cuda()
        optimizer.zero_grad()
        soh_pred, rul_pred = model(inputs)
        loss = criterion(soh_pred.squeeze(), soh_labels) + criterion(rul_pred.squeeze(), rul_labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{config["train"]["epochs"]}, Loss: {loss.item()}')

torch.save(model.state_dict(), '../models/battery_gpt_finetuned.pth')

import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from torch.cuda.amp import autocast

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the GRU Model (same as in training)
class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GRUNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# Load the trained model
input_size = 6  # Adjust this to match the number of features used in training
hidden_size = 50
output_size = 1
num_layers = 1

model = GRUNet(input_size, hidden_size, output_size, num_layers).to(device)

# Load the model weights
# 普通模型
# model.load_state_dict(torch.load('D:/ml/perflm/modelhdg/hemodel0903/global_model_round_20.pth'))

# dp模型
# model.load_state_dict(torch.load('D:/ml/perflm/modelhdg/hemodel0903dp/global_model_round_20.pth'))

# dpdqn模型
model.load_state_dict(torch.load('D:/ml/perflm/hemodel0903dpdqn2/global_model_round_19.pth'))

model.eval()

# Load and preprocess test data
def parse_line(line):
    try:
        json_str = line.split(' ', 1)[1].strip()
        data = json.loads(json_str)
        return (
            data['VP']['tst'],
            data['VP']['spd'],
            data['VP']['veh'],
            data['VP']['hdg'],
            data['VP']['lat'],
            data['VP']['long'],
            data['VP']['acc']
        )
    except:
        return None, None, None, None, None, None, None

data = []
with open('D:/ml/perflm/hedata0903/vehicle_108.txt', 'r') as file:
    for line in file:
        parsed_data = parse_line(line)
        if None not in parsed_data:
            data.append(parsed_data)

df = pd.DataFrame(data, columns=['timestamp', 'speed', 'veh', 'hdg', 'lat', 'long', 'acc'])
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['speed'] = df['speed'].astype(float)
df['veh'] = df['veh'].astype(int)
df['hdg'] = df['hdg'].astype(float)
df['lat'] = df['lat'].astype(float)
df['long'] = df['long'].astype(float)
df['acc'] = df['acc'].astype(float)
df = df.sort_values('timestamp')

scalers = {}
for column in ['speed', 'veh', 'hdg', 'lat', 'long', 'acc']:
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[column] = scaler.fit_transform(df[[column]])
    scalers[column] = scaler

def create_sequences(data, sequence_length):
    sequences = []
    targets = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i + sequence_length, :])
        targets.append(data[i + sequence_length, 2])  # Use index 2 for hdg instead of 0 for speed
    return np.array(sequences), np.array(targets)

sequence_length = 120
X, y = create_sequences(df[['speed', 'veh', 'hdg', 'lat', 'long', 'acc']].values, sequence_length)

X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=4, shuffle=False)  # Reduced batch size for memory efficiency

# Make predictions in batches
predictions = []

with torch.no_grad():
    with autocast():
        for X_batch in dataloader:
            X_batch = X_batch[0]  # Extract the features tensor
            batch_predictions = model(X_batch).cpu().numpy()
            predictions.append(batch_predictions)

predictions = np.concatenate(predictions, axis=0)

# Optionally, inverse transform predictions to the original scale
predictions_original_scale = scalers['hdg'].inverse_transform(predictions)

print(predictions_original_scale)

# Convert true values back to original scale
true_values_original_scale = scalers['hdg'].inverse_transform(y_tensor.cpu().numpy().reshape(-1, 1))

# Calculate Mean Squared Error (MSE)
mse = np.mean((predictions_original_scale - true_values_original_scale) ** 2)
print(f'Mean Squared Error: {mse}')

# Calculate Mean Absolute Error (MAE)
mae = np.mean(np.abs(predictions_original_scale - true_values_original_scale))
print(f'Mean Absolute Error: {mae}')

# Define a tolerance threshold (e.g., 5% of the true value)
tolerance = 0.1
# Calculate the difference between predictions and actual values
differences = np.abs(predictions_original_scale - true_values_original_scale)
# Calculate accuracy as the proportion of predictions within the tolerance
accuracy = np.mean((differences <= tolerance * true_values_original_scale).astype(float))
print(f'Accuracy: {accuracy * 100:.2f}%')

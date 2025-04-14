import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from torch.cuda.amp import GradScaler, autocast

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 1: Read and parse the data
def parse_line(line):
    try:
        # Extract the JSON part of the line
        json_str = line.split(' ', 1)[1].strip()
        data = json.loads(json_str)
        return data['VP']['tst'], data['VP']['spd']
    except:
        return None, None

# Define the GRU Model
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

# Function to prepare sequences
def create_sequences(data, sequence_length):
    sequences = []
    targets = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i+sequence_length])
        targets.append(data[i+sequence_length])
    return np.array(sequences), np.array(targets)

# Step 2: Load and process the test data
test_path = 'D:/ml/perflm/hedata0903/vehicle_76.txt'

data = []
with open(test_path, 'r') as file:
    for line in file:
        timestamp, speed = parse_line(line)
        if timestamp and speed is not None:
            data.append([timestamp, speed])

df = pd.DataFrame(data, columns=['timestamp', 'speed'])
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['speed'] = df['speed'].astype(float)
df = df.sort_values('timestamp')

min_max_scaler = MinMaxScaler(feature_range=(0, 1))
df['speed'] = min_max_scaler.fit_transform(df[['speed']])

sequence_length = 120  # Same as during training
X_test, y_test = create_sequences(df['speed'].values, sequence_length)

# Convert to PyTorch tensors and move to GPU
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

# Step 3: Load the trained global model
global_model = GRUNet(input_size=1, hidden_size=50, output_size=1, num_layers=1).to(device)
global_model.load_state_dict(torch.load('D:/ml/perflm/hemodel0903/global_model_round_50.pth'))  # Replace with the latest model checkpoint

# Step 4: Test the Model
global_model.eval()
with torch.no_grad():
    predictions = global_model(X_test_tensor).cpu().numpy()

# Step 5: Inverse transform predictions to the original scale
predictions_original_scale = min_max_scaler.inverse_transform(predictions)
true_values_original_scale = min_max_scaler.inverse_transform(y_test_tensor.cpu().numpy().reshape(-1, 1))

# Calculate Mean Squared Error (MSE)
mse = np.mean((predictions_original_scale - true_values_original_scale) ** 2)
print(f'Mean Squared Error: {mse}')

# Calculate Mean Absolute Error (MAE)
mae = np.mean(np.abs(predictions_original_scale - true_values_original_scale))
print(f'Mean Absolute Error: {mae}')

# Define a tolerance threshold (e.g., 5% of the true value)
tolerance = 0.05
# Calculate the difference between predictions and actual values
differences = np.abs(predictions_original_scale - true_values_original_scale)
# Calculate accuracy as the proportion of predictions within the tolerance
accuracy = np.mean((differences <= tolerance * true_values_original_scale).astype(float))
print(f'Accuracy: {accuracy * 100:.2f}%')

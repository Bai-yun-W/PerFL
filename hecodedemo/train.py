import json
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
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

data = []
with open('D:/ml/perflm/hedata0903/vehicle_20.txt', 'r') as file:
    for line in file:
        timestamp, speed = parse_line(line)
        if timestamp and speed is not None:
            data.append([timestamp, speed])

# Step 2: Preprocess the data
df = pd.DataFrame(data, columns=['timestamp', 'speed'])
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['speed'] = df['speed'].astype(float)

# Sort by timestamp
df = df.sort_values('timestamp')

# Normalize speed data
min_max_scaler = MinMaxScaler(feature_range=(0, 1))
df['speed'] = min_max_scaler.fit_transform(df[['speed']])

# Step 3: Prepare the dataset
def create_sequences(data, sequence_length):
    sequences = []
    targets = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i+sequence_length])
        targets.append(data[i+sequence_length])
    return np.array(sequences), np.array(targets)

sequence_length = 180  # Reduced sequence length to save memory
X, y = create_sequences(df['speed'].values, sequence_length)

# Convert to PyTorch tensors and move to GPU
X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(device)
y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

# Step 4: Create DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)  # Reduced batch size to save memory

# Step 5: Define the GRU Model
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

input_size = 1
hidden_size = 60  # Reduced hidden size to save memory
output_size = 1
num_layers = 1

model = GRUNet(input_size, hidden_size, output_size, num_layers).to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Adjusted learning rate
grad_scaler = GradScaler()  # Enable mixed precision

# Step 6: Train the Model
num_epochs = 200
model.train()

for epoch in range(num_epochs):
    for X_batch, y_batch in dataloader:
        optimizer.zero_grad()
        with autocast():
            output = model(X_batch)
            loss = criterion(output, y_batch.unsqueeze(1))
        grad_scaler.scale(loss).backward()

        # Apply gradient clipping to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        grad_scaler.step(optimizer)
        grad_scaler.update()
    
    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

# Step 7: Save the Model
torch.save(model.state_dict(), 'D:/ml/perflm/hemodel0903/gru_model_20.pth')

# After training, you can make predictions
model.eval()
with torch.no_grad():
    predictions = model(X_tensor).cpu().numpy()  # Move to CPU for further processing

# Optionally, inverse transform predictions to the original scale
predictions_original_scale = min_max_scaler.inverse_transform(predictions)

print(predictions_original_scale)

# Convert true values back to original scale
true_values_original_scale = min_max_scaler.inverse_transform(y_tensor.cpu().numpy().reshape(-1, 1))

# Calculate Mean Squared Error (MSE)
mse = np.mean((predictions_original_scale - true_values_original_scale) ** 2)
print(f'Mean Squared Error: {mse}')

# Calculate Mean Absolute Error (MAE)
mae = np.mean(np.abs(predictions_original_scale - true_values_original_scale))
print(f'Mean Absolute Error: {mae}')

# Define a tolerance threshold (e.g., 5% of the true value)
tolerance = 0.5
# Calculate the difference between predictions and actual values
differences = np.abs(predictions_original_scale - true_values_original_scale)
# Calculate accuracy as the proportion of predictions within the tolerance
accuracy = np.mean((differences <= tolerance * true_values_original_scale).astype(float))
print(f'Accuracy: {accuracy * 100:.2f}%')

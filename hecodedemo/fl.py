import json
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from torch.cuda.amp import GradScaler, autocast
import os

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

# Function to create sequences from data
def create_sequences(data, sequence_length):
    sequences = []
    targets = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i + sequence_length])
        targets.append(data[i + sequence_length])
    return np.array(sequences), np.array(targets)

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

# Function to train the model on a single client
def train_client(data_loader, model, criterion, optimizer, grad_scaler):
    model.train()
    for X_batch, y_batch in data_loader:
        optimizer.zero_grad()
        with autocast():
            output = model(X_batch)
            loss = criterion(output, y_batch.unsqueeze(1))
        grad_scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        grad_scaler.step(optimizer)
        grad_scaler.update()
    return model.state_dict()

# Function to perform federated averaging
def federated_averaging(global_model, client_states):
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_states[i][k].float() for i in range(len(client_states))], 0).mean(0)
    global_model.load_state_dict(global_dict)

# Step 2: Load and process data for multiple clients
client_data_paths = ['D:/ml/perflm/hedata0903/vehicle_20.txt', 
                     'D:/ml/perflm/hedata0903/vehicle_76.txt',
                     'D:/ml/perflm/hedata0903/vehicle_635.txt']

sequence_length = 120
batch_size = 8
hidden_size = 50
num_layers = 1
num_epochs = 10  # Number of epochs for each federated round
learning_rate = 0.0001

global_model = GRUNet(input_size=1, hidden_size=hidden_size, output_size=1, num_layers=num_layers).to(device)
criterion = nn.MSELoss().to(device)
grad_scaler = GradScaler()

# Perform federated learning
num_rounds = 50  # Number of federated rounds

for round in range(num_rounds):
    print(f"Round {round+1}/{num_rounds}")
    client_states = []

    for path in client_data_paths:
        data = []
        with open(path, 'r') as file:
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

        X, y = create_sequences(df['speed'].values, sequence_length)
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

        dataset = TensorDataset(X_tensor, y_tensor)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Load the global model weights to the client's model
        client_model = GRUNet(input_size=1, hidden_size=hidden_size, output_size=1, num_layers=num_layers).to(device)
        client_model.load_state_dict(global_model.state_dict())
        optimizer = optim.Adam(client_model.parameters(), lr=learning_rate)

        # Train the client model
        client_state = train_client(data_loader, client_model, criterion, optimizer, grad_scaler)
        client_states.append(client_state)

    # Perform federated averaging
    federated_averaging(global_model, client_states)

    # Save the global model after each round
    torch.save(global_model.state_dict(), f'D:/ml/perflm/hemodel0903/global_model_round_{round+1}.pth')

# After federated training, you can use the final global model for predictions
global_model.eval()
with torch.no_grad():
    predictions = global_model(X_tensor).cpu().numpy()

# Optionally, inverse transform predictions to the original scale
predictions_original_scale = min_max_scaler.inverse_transform(predictions)

print(predictions_original_scale)


import json
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from torch.cuda.amp import GradScaler, autocast

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Function to parse the data from the file
def parse_line(line):
    try:
        json_str = line.split(' ', 1)[1].strip()
        data = json.loads(json_str)
        return (
            data['VP']['tst'], data['VP']['spd'], data['VP']['veh'],
            data['VP']['hdg'], data['VP']['lat'], data['VP']['long'], data['VP']['acc']
        )
    except:
        return None, None, None, None, None, None, None

# Function to create sequences for training
def create_sequences(data, sequence_length):
    sequences = []
    targets = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i+sequence_length, :])
        targets.append(data[i+sequence_length, 2])  # Use index 2 for hdg
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

# Load the trained DQN model for dynamic noise adjustment
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load the DQN model
dqn_model_path = "D:/ml/perflm/dqn_vehicle_noise_model.pth"
state_size = 2  # Assuming veh_id and vehicle_type are the states
action_size = 10  # Number of discrete noise levels
qnetwork = QNetwork(state_size, action_size)
qnetwork.load_state_dict(torch.load(dqn_model_path))
qnetwork.eval()

# Discrete noise values
noise_values = np.linspace(0, 0.0002, action_size)

# Function to add Gaussian noise for differential privacy using DQN model
def add_noise_to_model(model, veh_id, vehicle_type):
    vehicle_type_numeric = 0 if vehicle_type == "bus" else 1
    state = torch.FloatTensor([veh_id, vehicle_type_numeric]).unsqueeze(0)

    with torch.no_grad():
        action_values = qnetwork(state)
    action = torch.argmax(action_values).item()
    noise_std = noise_values[action]
    print(noise_std)

    with torch.no_grad():
        for param in model.parameters():
            noise = torch.normal(0, noise_std, size=param.size()).to(param.device)
            param.add_(noise)

# Function to train the model on a single client
def train_client(data_loader, model, criterion, optimizer, grad_scaler, num_epochs, veh_id, vehicle_type):
    model.train()
    for epoch in range(num_epochs):
        for X_batch, y_batch in data_loader:
            optimizer.zero_grad()
            with autocast():
                output = model(X_batch)
                loss = criterion(output, y_batch.unsqueeze(1))
            grad_scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()
    
    # Add noise for differential privacy using DQN model
    add_noise_to_model(model, veh_id, vehicle_type)

    return model.state_dict()

# Function to perform federated averaging
def federated_averaging(global_model, client_states):
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_states[i][k].float() for i in range(len(client_states))], 0).mean(0)
    global_model.load_state_dict(global_dict)

# Paths to client data files
client_data_paths = ['D:/ml/perflm/hedata0903/vehicle_1.txt', 
                     'D:/ml/perflm/hedata0903/vehicle_2.txt',
                     'D:/ml/perflm/hedata0903/vehicle_3.txt',
                     'D:/ml/perflm/hedata0903/vehicle_4.txt',
                     'D:/ml/perflm/hedata0903/vehicle_5.txt',
                     'D:/ml/perflm/hedata0903/vehicle_6.txt',
                     'D:/ml/perflm/hedata0903/vehicle_109.txt',
                     'D:/ml/perflm/hedata0903/vehicle_110.txt',
                     'D:/ml/perflm/hedata0903/vehicle_111.txt', 
                     'D:/ml/perflm/hedata0903/vehicle_112.txt',
                     'D:/ml/perflm/hedata0903/vehicle_114.txt',
                     'D:/ml/perflm/hedata0903/vehicle_115.txt',
                     'D:/ml/perflm/hedata0903/vehicle_116.txt',
                     'D:/ml/perflm/hedata0903/vehicle_117.txt',
                     'D:/ml/perflm/hedata0903/vehicle_119.txt'
                    ]

sequence_length = 120
batch_size = 8
hidden_size = 50
num_layers = 1
num_epochs = 10  # Number of epochs for each federated round
learning_rate = 0.0001
num_rounds = 20  # Number of federated rounds

global_model = GRUNet(input_size=6, hidden_size=hidden_size, output_size=1, num_layers=num_layers).to(device)
criterion = nn.MSELoss().to(device)
grad_scaler = GradScaler()

# Perform federated learning
for round in range(num_rounds):
    print(f"Round {round+1}/{num_rounds}")
    client_states = []

    for path in client_data_paths:
        data = []
        with open(path, 'r') as file:
            for line in file:
                parsed_data = parse_line(line)
                if None not in parsed_data:
                    data.append(parsed_data)

        df = pd.DataFrame(data, columns=['timestamp', 'speed', 'veh', 'hdg', 'lat', 'long', 'acc'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df[['speed', 'veh', 'hdg', 'lat', 'long', 'acc']] = df[['speed', 'veh', 'hdg', 'lat', 'long', 'acc']].astype(float)
        df = df.sort_values('timestamp')

        veh_id = df['veh'].iloc[0]
        # print(veh_id)

        scalers = {}
        for column in ['speed', 'veh', 'hdg', 'lat', 'long', 'acc']:
            scaler = MinMaxScaler(feature_range=(0, 1))
            df[column] = scaler.fit_transform(df[[column]])
            scalers[column] = scaler

        X, y = create_sequences(df[['speed', 'veh', 'hdg', 'lat', 'long', 'acc']].values, sequence_length)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

        dataset = TensorDataset(X_tensor, y_tensor)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        client_model = GRUNet(input_size=6, hidden_size=hidden_size, output_size=1, num_layers=num_layers).to(device)
        client_model.load_state_dict(global_model.state_dict())
        optimizer = optim.Adam(client_model.parameters(), lr=learning_rate)

        vehicle_type = 'bus'  # 假设所有数据都是 bus 类型，或者从文件名或数据中提取实际类型

        client_state = train_client(data_loader, client_model, criterion, optimizer, grad_scaler, num_epochs, veh_id, vehicle_type)
        client_states.append(client_state)

    federated_averaging(global_model, client_states)

    # Save the global model after each round
    torch.save(global_model.state_dict(), f'D:/ml/perflm/hemodel0903dpdqn2/global_model_round_{round+1}.pth')

print("Training complete. Global model saved.")

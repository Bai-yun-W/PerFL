import torch
import torch.nn as nn
import numpy as np
import json
import random

# 定义与训练时相同的 Q-Network
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

# 加载模型
model_path = "D:/ml/perflm/dqn_vehicle_noise_model.pth"
state_size = 2
action_size = 10  # 动作空间大小与训练时相同
qnetwork = QNetwork(state_size, action_size)
qnetwork.load_state_dict(torch.load(model_path))
qnetwork.eval()

# 噪声离散值
noise_values = np.linspace(0, 0.0002, action_size)

# 定义测试函数
def test_model(veh_id, vehicle_type):
    vehicle_type_numeric = 0 if vehicle_type == "bus" else 1
    state = torch.FloatTensor([veh_id, vehicle_type_numeric]).unsqueeze(0)

    with torch.no_grad():
        action_values = qnetwork(state)
    action = torch.argmax(action_values).item()
    predicted_noise = noise_values[action]

    return predicted_noise

# 读取测试数据
file_path = 'D:/ml/perflm/all_vehicle_data_with_noise_0903.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    data = [json.loads(line) for line in file]

# 随机挑选30个车辆数据
sampled_data = random.sample(data, 30)

# 进行预测并输出结果
for record in sampled_data:
    veh_id = record["veh_id"]
    vehicle_type = record["vehicle_type"]
    actual_noise = record["noise"]

    predicted_noise = test_model(veh_id, vehicle_type)
    error = abs(predicted_noise - actual_noise)
    
    print(f"车辆编号: {veh_id}, 车辆类型: {vehicle_type}, 实际噪声: {actual_noise:.8f}, 预测噪声: {predicted_noise:.8f}, 误差: {error:.8f}")

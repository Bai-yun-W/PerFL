import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from torch.utils.data import Dataset, DataLoader
import json
import matplotlib.pyplot as plt

# 定义 Q-Network
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

# 定义数据集类
class VehicleDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                record = json.loads(line)
                veh_id = record["veh_id"]
                vehicle_type = 0 if record["vehicle_type"] == "bus" else 1  # 简单二分类
                noise = record["noise"]
                self.data.append((veh_id, vehicle_type, noise))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 环境类的简化版本
class VehicleEnv:
    def __init__(self, data):
        self.data = data
        self.current_idx = 0
        self.state_size = 2  # veh_id and vehicle_type
        self.action_size = 10  # Discretize noise values, e.g., [0.00000, 0.00002, 0.00004, ..., 0.00018]
        self.noise_values = np.linspace(0, 0.0002, self.action_size)

    def reset(self):
        self.current_idx = 0
        state = self.data[self.current_idx][:2]
        return np.array(state)

    def step(self, action):
        veh_id, vehicle_type = self.data[self.current_idx][:2]
        true_noise = self.data[self.current_idx][2]
        predicted_noise = self.noise_values[action]

        reward = -abs(true_noise - predicted_noise)  # Negative absolute error as reward
        self.current_idx += 1

        if self.current_idx >= len(self.data):
            done = True
            next_state = np.zeros(self.state_size)
        else:
            done = False
            next_state = np.array(self.data[self.current_idx][:2])

        return next_state, reward, done

# 训练 DQN 的主要函数
def train_dqn(env, num_episodes=1000, max_t=1000):
    state_size = env.state_size
    action_size = env.action_size
    qnetwork = QNetwork(state_size, action_size)
    optimizer = optim.Adam(qnetwork.parameters(), lr=0.0005)
    memory = deque(maxlen=50000)
    batch_size = 128
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995

    # 用于记录奖励的列表
    episode_rewards = []

    for i_episode in range(1, num_episodes + 1):
        state = env.reset()
        total_reward = 0
        for t in range(max_t):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            if random.random() > epsilon:
                with torch.no_grad():
                    action_values = qnetwork(state_tensor)
                action = torch.argmax(action_values).item()
            else:
                action = random.choice(np.arange(action_size))

            next_state, reward, done = env.step(action)
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            if done:
                break

            if len(memory) > batch_size:
                experiences = random.sample(memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*experiences)

                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones)

                q_values = qnetwork(states).gather(1, actions.unsqueeze(1)).squeeze()
                next_q_values = qnetwork(next_states).max(1)[0].detach()
                target_q_values = rewards + (gamma * next_q_values * (1 - dones))

                loss = nn.MSELoss()(q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(epsilon_min, epsilon_decay * epsilon)
        episode_rewards.append(total_reward)  # 记录当前回合的总奖励
        print(f"Episode {i_episode}/{num_episodes}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}")

    # 保存模型
    torch.save(qnetwork.state_dict(), "dqn_vehicle_noise_model.pth")
    print("DQN 训练完成并保存模型！")

    # 绘制收敛图
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(episode_rewards) + 1), episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('DQN Convergence')
    plt.grid()
    plt.savefig("dqn_convergence_plot.png")  # 保存收敛图
    plt.show()

# 使用经过训练的数据集
file_path = r"D:/ml/perflm/all_vehicle_data_with_noise_0903.txt"
dataset = VehicleDataset(file_path)
train_data = DataLoader(dataset, batch_size=len(dataset), shuffle=True)

for data in train_data:
    env = VehicleEnv(data)
    train_dqn(env)

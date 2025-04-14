import torch
import os
from collections import OrderedDict
from train import GRUNet

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 1: 加载所有模型的参数
model_dir = 'E:/mldata/hemodel0903'  # 模型保存的文件夹路径
model_files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith('.pth')]

# 检查模型文件是否存在
assert len(model_files) > 0, "未找到任何模型文件"

# 加载第一个模型参数，并移动到GPU上，以便初始化平均参数字典
average_state_dict = torch.load(model_files[0], map_location=device)

# 遍历并累加其余模型的参数
for model_file in model_files[1:]:
    state_dict = torch.load(model_file, map_location=device)
    for key in average_state_dict:
        average_state_dict[key] += state_dict[key]

# Step 3: 对累加的参数进行平均
for key in average_state_dict:
    average_state_dict[key] /= len(model_files)

# Step 4: 将平均后的参数保存到一个新的模型中
# 假设你的模型架构是 GRUNet 类
model = GRUNet(input_size=1, hidden_size=100, output_size=1, num_layers=1).to(device)
model.load_state_dict(average_state_dict)

# 保存平均后的模型参数
torch.save(model.state_dict(), 'E:/mldata/hemodel0903/gru_model_federated.pth')

print("模型参数已成功平均并保存")

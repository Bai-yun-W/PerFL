import random

# 生成10个0到1之间的随机数
random_numbers = [random.uniform(0, 2) for _ in range(15)]

# 输出生成的随机数
print(random_numbers)
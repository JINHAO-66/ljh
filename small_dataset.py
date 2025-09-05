import os
from PIL import Image
import random

# 类别
classes = ['kitchen_waste', 'recyclable', 'hazardous', 'other']
# 图片数量
num_images = 10  # 每类10张

# 创建目录
for split in ['train', 'val']:
    for cls in classes:
        path = os.path.join('data', split, cls)
        os.makedirs(path, exist_ok=True)

# 随机生成彩色图像
def generate_image(path):
    img = Image.new('RGB', (224, 224), 
                    (random.randint(0,255), random.randint(0,255), random.randint(0,255)))
    img.save(path)

# 生成训练和验证集
for split in ['train', 'val']:
    for cls in classes:
        for i in range(num_images):
            file_path = os.path.join('data', split, cls, f'{cls}_{i}.jpg')
            generate_image(file_path)

print("小样本数据集生成完成！每类10张图，共40张训练 + 40张验证。")

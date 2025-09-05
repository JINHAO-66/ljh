import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn as nn
import random

st.title("智能垃圾分类系统（演示版）")

# 初始化模型（随机权重，不需要训练）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_classes = 4
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model = model.to(device)
model.eval()  # 只用于演示，不训练

# 类别
classes = ['厨余垃圾','可回收物','有害垃圾','其他垃圾']

# 上传图片
uploaded_file = st.file_uploader("上传垃圾图片", type=['jpg','png','jpeg'])
if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='上传图片', use_column_width=True)

    # 随机预测（演示用）
    pred_class = random.choice(classes)
    st.success(f"预测类别：{pred_class}")

    # 如果你希望用模型做推理，也可以启用下面代码（结果仍然随机，因为模型未训练）
    """
    preprocess = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, pred = torch.max(outputs, 1)
    st.info(f"模型预测类别：{classes[pred.item()]}")
    """

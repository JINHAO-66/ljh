import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn as nn

st.title("智能垃圾分类系统")

# 加载模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_classes = 4
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model.load_state_dict(torch.load('model/model.pth', map_location=device))
model = model.to(device)
model.eval()

# 类别
classes = ['厨余垃圾','可回收物','有害垃圾','其他垃圾']

# 上传图片
uploaded_file = st.file_uploader("上传垃圾图片", type=['jpg','png','jpeg'])
if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='上传图片', use_column_width=True)

    # 预处理
    preprocess = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    # 预测
    with torch.no_grad():
        outputs = model(input_tensor)
        _, pred = torch.max(outputs, 1)
    st.success(f"预测类别：{classes[pred.item()]}")

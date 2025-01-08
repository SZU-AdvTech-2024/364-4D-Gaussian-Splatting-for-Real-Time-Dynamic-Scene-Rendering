import torch
import cv2
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# 加载预训练模型
model_type = "DPT_Large"  # 可选: "MiDaS_small", "DPT_Hybrid"
model = torch.hub.load("intel-isl/MiDaS", model_type)
model.eval()

# 预处理图像
transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
demo_img='/mnt/d/yunfei-174/3DGS/data/images/image_c_000_f_000000.png'
img = cv2.imread(demo_img)
input_batch = transform(img)

# 推理深度图
with torch.no_grad():
    prediction = model(input_batch)

# 后处理并保存结果
depth_map = prediction.squeeze().cpu().numpy()
depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())  # 归一化
cv2.imwrite("depth_map.png", (depth_map * 255).astype("uint8"))

import numpy as np
import matplotlib.pyplot as plt
import cv2

# 创建黑色背景图像
image = np.zeros((256, 256, 3), dtype=np.uint8)
image.fill(0)  # 将图像填充为黑色

# 定义关键点坐标
# x_coords = [34, 73, 83, 147, 203, 73, 138, 194, 196, -1, -1, 194, -1, -1, -1, 26, -1, 27]
# y_coords = [62, 98, 132, 131, 129, 64, 45, 35, 106, -1, -1, 62, -1, -1, -1, 67, -1, 87]
x_coords, y_coords = [32, 56, 64, 102, 125, 48, 84, 112, 113, 137, 181, 115, 144, 175, 29, 26, 34, -1],[97, 87, 67, 56, 34, 107, 131, 105, 80, 33, 69, 105, 68, 96, 90, 97, 76, -1]

# 定义连接顺序和颜色列表
LIMB_SEQ = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7], [1,8], [8,9], [9,10], [1,11], [11,12], [12,13], [1,0], [0,14], [14,16], [0,15], [15,17], [2,16], [5,17]]
COLORS = [[1.0, 0.0, 0.0], [1.0, 0.333, 0.0], [1.0, 0.667, 0.0], [1.0, 1.0, 0.0], [0.667, 1.0, 0.0], [0.333, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.333],
          [0.0, 1.0, 0.667], [0.0, 1.0, 1.0], [0.0, 0.667, 1.0], [0.0, 0.333, 1.0], [0.0, 0.0, 1.0], [0.333, 0.0, 1.0], [0.667, 0.0, 1.0], [1.0, 0.0, 1.0],
          [1.0, 0.0, 0.667], [1.0, 0.0, 0.333]]

# 绘制关键点
for i, (x, y) in enumerate(zip(x_coords, y_coords)):
    if x != -1 and y != -1:  # 仅当坐标不为-1时绘制关键点
        color = tuple(int(c * 255) for c in COLORS[i])  # 将颜色值转换为整数
        cv2.circle(image, (x, y), 2, color, -1)  # 使用color绘制关键点，半径为2

# 绘制连接线
for limb in LIMB_SEQ:
    x1, y1 = x_coords[limb[0]], y_coords[limb[0]]
    x2, y2 = x_coords[limb[1]], y_coords[limb[1]]
    if x1 != -1 and y1 != -1 and x2 != -1 and y2 != -1:  # 仅当两个关键点坐标都不为-1时绘制连接线
        image = cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 1)  # 使用白色绘制连接线

# 逆时针旋转90度
image = np.rot90(image, k=3)

image = image[:, 80:256, :]

image = cv2.flip(image, 1)
# 保存图像
save_path = r"pose-picture\fashionMENSweatshirts_Hoodiesid0000088102_4full.jpg"
cv2.imwrite(save_path, image)
# 显示图像
plt.imshow(image)
plt.axis('off')  # 隐藏坐标轴
plt.show()
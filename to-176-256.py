from PIL import Image
import os

# 源文件夹路径
source_folder = "GFLA_MY"
# 目标文件夹路径
target_folder = "GFLA_MY_176_256"

# 创建目标文件夹
os.makedirs(target_folder, exist_ok=True)

# 遍历源文件夹中的所有文件
for filename in os.listdir(source_folder):
    # 拼接源文件路径
    source_path = os.path.join(source_folder, filename)
    # 拼接目标文件路径
    target_path = os.path.join(target_folder, filename)

    # 打开图像文件
    image = Image.open(source_path)

    # 裁剪图像
    # cropped_image = image.crop((40, 0, 216, 256))

    # 调整图像分辨率
    # resized_image = cropped_image.resize((176, 256))
    resized_image = image.resize((176, 256))

    # 保存调整后的图像到目标文件夹
    resized_image.save(target_path)
    print(target_path,'is ok')

    # 关闭图像文件
    image.close()

print("图片处理完成！")
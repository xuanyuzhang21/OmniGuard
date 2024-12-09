from PIL import Image
import numpy as np

def save_residual_image(image1_path, image2_path, output_path):
    # 打开两张图片
    img1 = Image.open(image1_path)
    img2 = Image.open(image2_path)

    # 将图片转换为灰度图（如果它们是彩色图像）
    img1 = img1.convert('RGB')
    img2 = img2.convert('RGB')

    # 将图片转换为 numpy 数组
    img1_array = np.array(img1)
    img2_array = np.array(img2)

    # 确保两张图片大小一致
    if img1_array.shape != img2_array.shape:
        print("图片尺寸不一致，请确保两张图片大小相同。")
        return

    # 计算残差图（绝对差）
    residual_image_array = np.abs(img1_array - img2_array)

    # 将残差图转换回 PIL 图像
    residual_image = Image.fromarray(residual_image_array)

    # 保存残差图
    residual_image.save(output_path)

    print(f"残差图已保存为 '{output_path}'")

# 示例调用
save_residual_image('/data03/zxy/OmniGuard/111.png', '/data03/zxy/OmniGuard/ed5716a2d2e6f07b1a01d11628fe6cf8.jpeg', 'residual_image.png')


# import cv2
# import numpy as np
# from PIL import Image

# # 读取两张图片
# # image1 = cv2.imread('/data03/zxy/watermark-anything/assets/images/bluesky.png', cv2.IMREAD_GRAYSCALE)
# # image2 = cv2.imread('/data03/zxy/watermark-anything/outputs/bluesky.png_wm.png', cv2.IMREAD_GRAYSCALE)
# # print(image1.min(), image2.max())

# def calculate_psnr(image1_path, image2_path):
#     # 打开图片并转换为灰度图
#     img1 = Image.open(image1_path).convert('RGB')  # 'L' 模式表示灰度图
#     img2 = Image.open(image2_path).convert('RGB')

#     # 将图片转换为 numpy 数组
#     img1 = np.array(img1)
#     img2 = np.array(img2)

#     # 确保图片大小相同
#     if img1.shape != img2.shape:
#         print("图片尺寸不一致，请确保两张图片大小相同。")
#         return None

#     # 计算均方误差 (MSE)
#     mse = np.mean((img1 - img2) ** 2)
    
#     if mse == 0:
#         return 100  # 如果两张图完全相同，PSNR 为无穷大，可以设为100

#     # 计算 PSNR
#     max_pixel = 255.0  # 对于 8 位图像，最大像素值是 255
#     psnr = 10 * np.log10((max_pixel ** 2) / mse)
    
#     return psnr

# print(calculate_psnr('/data03/zxy/OmniGuard/111.png', '/data03/zxy/OmniGuard/ed5716a2d2e6f07b1a01d11628fe6cf8.jpeg'))


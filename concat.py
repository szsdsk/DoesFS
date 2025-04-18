from PIL import Image
import os

def concat_images_horizontally(image_paths, output_path):
    # 加载所有图片
    images = [Image.open(p) for p in image_paths]

    # 确保所有图片的高度一致（可以根据需要进行 resize）
    min_height = min(img.height for img in images)
    images = [img.resize((int(img.width * min_height / img.height), min_height)) for img in images]

    # 计算输出图像的宽度和高度
    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)

    # 创建一张新的空白图像
    new_image = Image.new('RGB', (total_width, max_height))

    # 拼接图片
    x_offset = 0
    for img in images:
        new_image.paste(img, (x_offset, 0))
        x_offset += img.width

    # 保存结果
    new_image.save(output_path)
    print(f"拼接完成，保存为 {output_path}")


path = 'data/style_images_aligned'
image = '009'
# 示例用法
# image_files = ['target1.png', 'target2.png', 'target3.png', 'target4.png', 'target5.png', 'target6.png', 'target7.png']
image_files = [f'images/MTG_pair/target{i}/{image}.png' for i in [1, 3, 5, 6]]
# image_files = ['003.png', '006.png', '009.png', '010.png', '011.png']
# image_files = [os.path.join(path, p) for p in image_files]
output_file = 'outputs/concat/21.png'
concat_images_horizontally(image_files, output_file)

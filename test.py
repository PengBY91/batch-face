import cv2
import numpy as np
import os
import re
from collections import defaultdict
from batch_face import RetinaFace

imgs_folder = 'examples/杜宏 热玛吉照片test 1025/'

def parse_filename(filename):
    """
    解析文件名，提取日期和序号
    格式: "日期(序号).jpg" 或 "日期（序号）.jpg"
    例如: "20250111(1).jpg" 或 "20250111 (1).jpg" 或 "20250111( 1 ).jpg"
    返回: (日期字符串, 序号) 或 (None, None)
    """
    # 匹配模式: 日期 [可选空格] [英文或中文括号] [可选空格] 序号 [可选空格] [英文或中文括号] [可选空格] .扩展名
    # 支持英文括号 () 和中文括号 （）
    # \s* 匹配0个或多个空白字符（包括空格、制表符等）
    pattern = r'(\d{8})\s*[\(（]\s*(\d+)\s*[\)）]\s*\.'
    match = re.search(pattern, filename)
    if match:
        date_str = match.group(1)
        seq_num = int(match.group(2))
        return date_str, seq_num
    return None, None

def calculate_yaw_angle(kps):
    """
    根据5个关键点计算头部左右转动的角度（Yaw角度）
    kps: 5个关键点 [左眼, 右眼, 鼻子, 左嘴角, 右嘴角]
    返回: yaw角度（单位：度）
    """
    left_eye = kps[0]
    right_eye = kps[1]
    nose = kps[2]
    
    # 计算两眼中点
    eye_center = (left_eye + right_eye) / 2
    
    # 计算两眼之间的距离（作为参考尺度）
    eye_distance = np.linalg.norm(right_eye - left_eye)
    
    # 计算 Yaw 角度（左右转头）
    # 方法：比较鼻子相对于两眼中点的水平偏移
    # 当头转向左侧时，鼻子会向左偏移（x减小）
    # 当头转向右侧时，鼻子会向右偏移（x增大）
    nose_offset_x = nose[0] - eye_center[0]
    
    # 归一化偏移量并转换为角度
    # 假设鼻子在两眼中点正下方时为0度
    # 经验系数：完全侧脸时鼻子偏移约为眼距的一半
    yaw = -np.degrees(np.arctan2(nose_offset_x, eye_distance)) * 2.5
    
    return yaw

def crop_face_with_expansion(img, box, scale=1.5):
    """
    按照中心点不变，放大 bounding box 并截取人脸区域
    """
    x1, y1, x2, y2 = box.astype(int)
    
    # 计算中心点
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # 计算原始宽高
    width = x2 - x1
    height = y2 - y1
    
    # 放大
    new_width = width * scale
    new_height = height * scale
    
    # 计算放大后的边界框（中心点不变）
    new_x1 = int(center_x - new_width / 2)
    new_y1 = int(center_y - new_height / 2)
    new_x2 = int(center_x + new_width / 2)
    new_y2 = int(center_y + new_height / 2)
    
    # 确保不超出图像边界
    img_h, img_w = img.shape[:2]
    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    new_x2 = min(img_w, new_x2)
    new_y2 = min(img_h, new_y2)
    
    # 截取放大后的人脸区域
    face_crop = img[new_y1:new_y2, new_x1:new_x2].copy()
    
    return face_crop, (new_x1, new_y1, new_x2, new_y2)

def crop_face_with_target_ratio(img, box, scale=1.5, target_ratio=None):
    """
    按照目标宽高比裁剪人脸区域（中心点不变）
    
    参数:
        img: 输入图像
        box: 人脸检测框 [x1, y1, x2, y2]
        scale: 放大倍数
        target_ratio: 目标宽高比 (width/height)，如果为 None 则使用原始比例
    
    返回:
        face_crop: 裁剪后的人脸图像
        box_info: 裁剪框信息 (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = box.astype(int)
    
    # 计算中心点
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # 计算原始宽高
    width = x2 - x1
    height = y2 - y1
    
    # 放大
    new_width = width * scale
    new_height = height * scale
    
    # 如果指定了目标宽高比，调整裁剪区域
    if target_ratio is not None:
        current_ratio = new_width / new_height
        
        if current_ratio > target_ratio:
            # 当前比例偏宽，需要增加高度
            new_height = new_width / target_ratio
        else:
            # 当前比例偏高，需要增加宽度
            new_width = new_height * target_ratio
    
    # 计算调整后的边界框（中心点不变）
    new_x1 = int(center_x - new_width / 2)
    new_y1 = int(center_y - new_height / 2)
    new_x2 = int(center_x + new_width / 2)
    new_y2 = int(center_y + new_height / 2)
    
    # 确保不超出图像边界
    img_h, img_w = img.shape[:2]
    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    new_x2 = min(img_w, new_x2)
    new_y2 = min(img_h, new_y2)
    
    # 截取调整后的人脸区域
    face_crop = img[new_y1:new_y2, new_x1:new_x2].copy()
    
    return face_crop, (new_x1, new_y1, new_x2, new_y2)

def place_face_on_canvas(face_img, canvas_w, canvas_h, bg_color=(255, 255, 255)):
    """
    将人脸直接缩放到画布尺寸，填满整个画布（不留空白）
    由于所有人脸都是按照统一宽高比裁剪的，直接缩放不会明显变形
    """
    # 直接缩放到画布尺寸，填满整个画布
    resized = cv2.resize(face_img, (canvas_w, canvas_h), interpolation=cv2.INTER_LINEAR)
    
    return resized

def process_folder(folder_path, detector, max_size, resize, threshold, output_folder='examples'):
    """
    处理单个文件夹中的所有图像
    """
    folder_name = os.path.basename(folder_path)
    print(f"\n{'#'*70}")
    print(f"# 开始处理文件夹: {folder_name}")
    print(f"# 路径: {folder_path}")
    print(f"{'#'*70}")
    
    # 获取文件夹中的所有图像文件并按日期分组
    # 支持的图像格式（不区分大小写）: jpg, JPG, jpeg, JPEG, png, PNG, bmp, BMP
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    date_groups = defaultdict(list)  # 按日期分组

    for filename in os.listdir(folder_path):
        ext = os.path.splitext(filename)[1].lower()  # 转换为小写进行比较
        if ext in image_extensions:
            date_str, seq_num = parse_filename(filename)
            if date_str:
                date_groups[date_str].append({
                    'path': os.path.join(folder_path, filename),
                    'filename': filename,
                    'seq_num': seq_num
                })
            else:
                print(f"    警告：文件名格式不符合规范，跳过: {filename}")

    if len(date_groups) == 0:
        print(f"  ❌ 文件夹 {folder_name} 中未找到符合格式的图像文件")
        return None

    # 按日期排序（时间最早的在前）
    sorted_dates = sorted(date_groups.keys())
    print(f"\n找到 {len(sorted_dates)} 个日期的数据:")
    for date in sorted_dates:
        print(f"  {date}: {len(date_groups[date])} 张图像")

    # 第一步：检测所有人脸，收集基本信息（用于计算目标宽高比）
    print(f"\n{'='*60}")
    print(f"第一遍扫描：收集所有人脸信息...")
    print(f"{'='*60}")
    
    all_face_info = []  # 存储所有人脸的基本信息
    
    for date in sorted_dates:
        print(f"\n处理日期: {date}")
        
        # 处理该日期的所有图像
        for img_info in date_groups[date]:
            img_path = img_info['path']
            filename = img_info['filename']
            
            print(f"  检测: {filename}", end=' ')
            
            img = cv2.imread(img_path)
            if img is None:
                print(f"- 无法读取")
                continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 检测人脸
            faces = detector(img_rgb, threshold=threshold, resize=resize, max_size=max_size, return_dict=True)
            
            if len(faces) == 0:
                print(f"- 未检测到人脸")
                continue
            
            # 只保留置信度最高的人脸
            if len(faces) > 1:
                best_face = max(faces, key=lambda f: f['score'])
            else:
                best_face = faces[0]
            
            # 计算初步裁剪后的宽高比
            face_crop, _ = crop_face_with_expansion(img, best_face['box'], scale=1.5)
            h, w = face_crop.shape[:2]
            ratio = w / h
            
            all_face_info.append({
                'date': date,
                'img_path': img_path,
                'filename': filename,
                'img': img,
                'face': best_face,
                'ratio': ratio
            })
            
            print(f"- 成功 (宽高比: {ratio:.3f})")
    
    if len(all_face_info) == 0:
        print(f"\n❌ 文件夹 {folder_name} 中没有任何有效的人脸数据")
        return None
    
    # 计算目标宽高比（使用中位数，更稳健）
    all_ratios = [info['ratio'] for info in all_face_info]
    target_ratio = np.median(all_ratios)
    
    print(f"\n{'='*60}")
    print(f"统计信息:")
    print(f"  总人脸数: {len(all_face_info)}")
    print(f"  宽高比范围: {min(all_ratios):.3f} ~ {max(all_ratios):.3f}")
    print(f"  目标宽高比（中位数）: {target_ratio:.3f}")
    print(f"{'='*60}")

    # 第二步：按照目标宽高比重新裁剪所有人脸
    print(f"\n{'='*60}")
    print(f"第二遍扫描：按统一宽高比裁剪人脸...")
    print(f"{'='*60}")
    
    all_faces_by_date = {}  # 存储每个日期的人脸数据
    
    for info in all_face_info:
        date = info['date']
        img = info['img']
        best_face = info['face']
        filename = info['filename']
        
        # 计算 yaw 角度
        yaw = calculate_yaw_angle(best_face['kps'])
        
        # 按照目标宽高比裁剪人脸
        face_crop, expanded_box = crop_face_with_target_ratio(
            img, best_face['box'], scale=1.5, target_ratio=target_ratio
        )
        
        if date not in all_faces_by_date:
            all_faces_by_date[date] = []
        
        all_faces_by_date[date].append({
            'image': face_crop,
            'yaw': yaw,
            'score': best_face['score'],
            'filename': filename
        })
    
    # 对每个日期的人脸按 yaw 角度排序
    for date in all_faces_by_date:
        all_faces_by_date[date].sort(key=lambda x: x['yaw'], reverse=True)
        
        print(f"\n日期 {date} ({len(all_faces_by_date[date])} 个人脸):")
        for i, data in enumerate(all_faces_by_date[date]):
            h, w = data['image'].shape[:2]
            actual_ratio = w / h
            print(f"  {i+1}. {data['filename'][:20]:20s} - Yaw: {data['yaw']:.2f}° - 尺寸: {w}x{h} - 比例: {actual_ratio:.3f}")

    # 第三步：设置统一的画布尺寸（根据目标宽高比计算，确保无空白）
    print(f"\n{'='*60}")
    print(f"设置统一画布尺寸...")
    print(f"{'='*60}")

    # 收集所有人脸的尺寸信息
    all_faces_list = []
    for date, faces in all_faces_by_date.items():
        all_faces_list.extend(faces)

    total_faces = len(all_faces_list)
    print(f"总人脸数: {total_faces}")

    # 根据目标宽高比设置画布尺寸，确保人脸能填满整个画布（无空白）
    canvas_width = 400
    canvas_height = int(canvas_width / target_ratio)  # 根据目标宽高比计算高度

    print(f"所有人脸的尺寸范围:")
    print(f"  宽度: {min(f['image'].shape[1] for f in all_faces_list)} ~ {max(f['image'].shape[1] for f in all_faces_list)}")
    print(f"  高度: {min(f['image'].shape[0] for f in all_faces_list)} ~ {max(f['image'].shape[0] for f in all_faces_list)}")
    print(f"统一画布尺寸: {canvas_width}x{canvas_height} (宽高比: {canvas_width/canvas_height:.3f} = 目标宽高比: {target_ratio:.3f})")

    # 第四步：将所有人脸按比例缩放并居中放置在统一画布上
    print(f"\n{'='*60}")
    print(f"调整人脸到统一画布并拼接...")
    print(f"{'='*60}")

    date_rows = []
    for date in sorted_dates:
        if date not in all_faces_by_date:
            continue
        
        date_face_data = all_faces_by_date[date]
        
        # 将该日期的所有人脸放置到统一画布上
        canvas_faces = []
        for data in date_face_data:
            img = data['image']
            # 保持宽高比，居中放置在画布上
            canvas_img = place_face_on_canvas(img, canvas_width, canvas_height)
            canvas_faces.append(canvas_img)
        
        # 水平拼接该日期的所有人脸
        row_image = np.hstack(canvas_faces)
        date_rows.append({
            'date': date,
            'image': row_image,
            'count': len(date_face_data)
        })
        
        print(f"日期 {date}: {len(date_face_data)} 个人脸 -> 拼接行尺寸: {row_image.shape[1]}x{row_image.shape[0]}")

    # 垂直拼接所有日期的行（行与行之间添加间隔，并显示日期）
    spacing_color = (240, 240, 240)  # 浅灰色分隔条
    text_color = (80, 80, 80)  # 深灰色文字
    
    if len(date_rows) > 0:
        # 创建带间隔的行列表
        rows_with_spacing = []
        row_width = date_rows[0]['image'].shape[1]
        
        for i, row_data in enumerate(date_rows):
            rows_with_spacing.append(row_data['image'])
            # 在最后一行之后不添加间隔
            if i < len(date_rows) - 1:
                # 在分隔条上添加下一行的日期信息
                next_date = date_rows[i + 1]['date']
                # 格式化日期：20250111 -> 2025-01-11
                formatted_date = f"{next_date[:4]}-{next_date[4:6]}-{next_date[6:8]}"
                date_text = f"日期: {formatted_date}"
                
                # 设置字体参数
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.0
                thickness = 2
                
                # 计算文本尺寸
                (text_width, text_height), baseline = cv2.getTextSize(date_text, font, font_scale, thickness)
                
                # 分隔条高度仅为文字高度加上很小的padding（上下各2px）
                padding = 4  # 上下各2px
                row_spacing = text_height + baseline + padding
                
                # 创建分隔条（高度刚好容纳文字）
                spacer = np.ones((row_spacing, row_width, 3), dtype=np.uint8) * np.array(spacing_color, dtype=np.uint8)
                
                # 计算文本位置（居中）
                text_x = (row_width - text_width) // 2
                text_y = text_height + padding // 2  # 从上边距开始，加上文字高度
                
                # 在分隔条上绘制日期文本
                cv2.putText(spacer, date_text, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)
                
                rows_with_spacing.append(spacer)
        
        # 垂直拼接
        final_image = np.vstack(rows_with_spacing)
    else:
        final_image = None
    
    print(f"\n最终拼接图像尺寸: {final_image.shape[1]}x{final_image.shape[0]}")
    print(f"总计: {len(date_rows)} 个日期, {sum(row['count'] for row in date_rows)} 个人脸")
    print(f"每个人脸画布尺寸: {canvas_width}x{canvas_height} (填满整个画布，无空白)")
    print(f"统一宽高比: {target_ratio:.3f} (所有子图比例一致)")
    print(f"行间距: 仅日期文字高度 (动态计算)")

    # 保存拼接结果
    output_path = os.path.join(output_folder, f'concatenated_{folder_name}.jpg')
    cv2.imwrite(output_path, final_image)
    print(f"\n✅ 结果已保存到: {output_path}")
    
    return output_path

# 主程序
if __name__ == '__main__':
    # 初始化检测器
    detector = RetinaFace(device='cpu')
    
    max_size = 1080
    resize = 1
    threshold = 0.75
    
    if not os.path.exists(imgs_folder):
        print(f"错误：文件夹 {imgs_folder} 不存在")
        exit()
    
    # 获取所有子文件夹
    subfolders = []
    for item in os.listdir(imgs_folder):
        item_path = os.path.join(imgs_folder, item)
        if os.path.isdir(item_path):
            subfolders.append(item_path)
    
    if len(subfolders) == 0:
        # 如果没有子文件夹，则直接处理当前文件夹
        print("未找到子文件夹，直接处理当前文件夹...")
        process_folder(imgs_folder, detector, max_size, resize, threshold)
    else:
        # 处理所有子文件夹
        print(f"\n{'*'*70}")
        print(f"* 找到 {len(subfolders)} 个子文件夹，将依次处理")
        print(f"{'*'*70}")
        
        results = []
        for folder_path in subfolders:
            result = process_folder(folder_path, detector, max_size, resize, threshold)
            if result:
                results.append(result)
        
        print(f"\n{'*'*70}")
        print(f"* 处理完成！")
        print(f"* 成功处理 {len(results)} / {len(subfolders)} 个文件夹")
        print(f"{'*'*70}")
        
        if results:
            print("\n生成的文件:")
            for i, path in enumerate(results, 1):
                print(f"  {i}. {path}")

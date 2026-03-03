import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import os
import platform

output_dir = "synthetic_imgs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def get_chinese_font(size=20):

    system = platform.system()
    font_path = ""
    
    if system == "Windows":

        candidates = [
            r"C:\Windows\Fonts\simhei.ttf",      # 黑体
            r"C:\Windows\Fonts\msyh.ttc",        # 微软雅黑
            r"C:\Windows\Fonts\simsun.ttc",      # 宋体
        ]
        for path in candidates:
            if os.path.exists(path):
                font_path = path
                break
    
    elif system == "Darwin": # macOS
        candidates = [
            "/System/Library/Fonts/PingFang.ttc",
            "/Library/Fonts/Arial Unicode.ttf"
        ]
        for path in candidates:
            if os.path.exists(path):
                font_path = path
                break
                
    elif system == "Linux":
        # Linux 通常需要手动安装字体，这里列举常见位置
        candidates = [
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", # 文泉驿正黑
            "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc"
        ]
        for path in candidates:
            if os.path.exists(path):
                font_path = path
                break

    # 如果找到了系统字体，加载它；否则尝试加载当前目录下的字体或抛出提示
    if font_path and os.path.exists(font_path):
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            pass
            
    # 尝试加载当前目录下的字体文件
    local_fonts = ["simhei.ttf", "msyh.ttc", "Arial.Unicode.ttf"]
    for lf in local_fonts:
        if os.path.exists(lf):
            return ImageFont.truetype(lf, size)
            
    # 如果实在找不到，返回默认字体（但这会导致中文乱码或报错，所以最好打印警告）
    print("警告：未找到合适的中文字体，中文内容可能无法正确显示。请确保系统安装了中文字体或将字体文件放在脚本同级目录。")
    return ImageFont.load_default()

def create_base_report(data):
    """渲染一份基础的化验单文本图"""
    # 创建 A4 比例的画布
    width, height = 800, 1100
    img = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # 【修复核心】加载支持中文的字体
    # 标题稍大，正文稍小
    font_title = get_chinese_font(size=24)
    font_text = get_chinese_font(size=18)
    
    # 为了防止字体加载失败导致后续崩溃，做个简单检查，如果还是默认字体且包含中文，可能会再次报错，
    # 但在 get_chinese_font 中我们已经尽量寻找系统字体了。

    # 绘制页眉
    meta = data['report_metadata']
    # 使用 font_title 绘制中文
    draw.text((300, 40), meta['institution'], fill=(0,0,0), font=font_title)
    draw.text((280, 70), meta['report_type'], fill=(0,0,0), font=font_title)
    draw.line((40, 110, 760, 110), fill=(0,0,0), width=2)
    
    # 绘制病人信息
    info = meta['patient_info']
    info_str = f"姓名: {info['name']}  性别: {info['sex']}  年龄: {info['age']}  病区: {info['ward']}"
    draw.text((50, 130), info_str, fill=(0,0,0), font=font_text)
    draw.text((50, 155), f"临床诊断: {meta['clinical_diagnosis']}", fill=(0,0,0), font=font_text)
    draw.text((550, 130), f"ID: {info['clinical_id']}", fill=(0,0,0), font=font_text)
    draw.line((40, 185, 760, 185), fill=(0,0,0), width=1)

    # 绘制表格表头
    draw.text((50, 200), "项目名称", fill=(0,0,0), font=font_text)
    draw.text((250, 200), "缩写", fill=(0,0,0), font=font_text)
    draw.text((400, 200), "结果", fill=(0,0,0), font=font_text)
    draw.text((550, 200), "参考范围", fill=(0,0,0), font=font_text)
    draw.text((700, 200), "单位", fill=(0,0,0), font=font_text)

    # 绘制结果
    y = 230
    for item in data['test_results']:
        # 简单判断是否异常
        try:
            val = float(item['value'])
            # 处理范围字符串，防止格式错误
            if ' ~ ' in item['range']:
                low, high = map(float, item['range'].split(' ~ '))
            else:
                # 兼容某些可能没有范围的情况，设为极大极小值避免报错
                low, high = -9999, 9999
            
            is_abnormal = val < low or val > high
            
            color = (0, 0, 0)
            marker = ""
            if is_abnormal:
                marker = " ↑" if val > high else " ↓"
            
            draw.text((50, y), item['name'], fill=color, font=font_text)
            draw.text((250, y), item['abbreviation'], fill=color, font=font_text)
            draw.text((400, y), f"{item['value']}{marker}", fill=color, font=font_text)
            draw.text((550, y), item['range'], fill=color, font=font_text)
            draw.text((700, y), item['unit'], fill=color, font=font_text)
            y += 35
        except ValueError:
            # 如果数值转换失败，跳过该项或打印错误，防止程序中断
            continue
        
    draw.line((40, 950, 760, 950), fill=(0,0,0), width=1)
    draw.text((50, 970), f"检测时间: {meta['test_date']}", fill=(0,0,0), font=font_text)
    draw.text((550, 970), "检验医师: AI-Agent", fill=(0,0,0), font=font_text)
    
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# ... 后面的 apply_augmentation 函数保持不变 ...
def apply_augmentation(img, variant_id):
    """核心：执行架构文档 3.2 的物理增强"""
    h, w = img.shape[:2]
    
    if "Var_01" in variant_id: 
        return cv2.GaussianBlur(img, (5, 5), 3)
    
    elif "Var_02" in variant_id: 
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(mask, (w//2, h//3), 400, 255, -1)
        mask = cv2.GaussianBlur(mask, (501, 501), 200) / 255.0
        img = img.astype(np.float32)
        for i in range(3): img[:,:,i] += mask * 150
        return np.clip(img, 0, 255).astype(np.uint8)
    elif "Var_03" in variant_id: 
        return cv2.convertScaleAbs(img, alpha=0.6, beta=-30)
    elif "Var_04" in variant_id: 
        pts1 = np.float32([[0,0],[w,0],[0,h],[w,h]])
        pts2 = np.float32([[w*0.1, h*0.05], [w*0.9, 0], [0, h], [w, h*0.95]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        return cv2.warpPerspective(img, M, (w, h), borderValue=(255,255,255))
    elif "Var_05" in variant_id: 
        kernel = np.zeros((5, 5))
        kernel[2, :] = 1.0 / 5.0
        return cv2.filter2D(img, -1, kernel)
    elif "Var_06" in variant_id: 
        rows, cols = img.shape[:2]
        img_output = np.zeros(img.shape, dtype=img.dtype)
        for i in range(rows):
            for j in range(cols):
                offset_x = int(5.0 * np.sin(2 * np.pi * i / 180))
                if j+offset_x < cols:
                    img_output[i,j] = img[i,(j+offset_x)%cols]
                else:
                    img_output[i,j] = 255
        return img_output
    elif "Var_07" in variant_id: 
        overlay = img.copy()
        cv2.ellipse(overlay, (200, 500), (100, 150), 30, 0, 360, (150, 200, 230), -1)
        return cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
    elif "Var_08" in variant_id: 
        cv2.circle(img, (420, 790), 40, (0, 0, 255), 3)
        return img
    elif "Var_09" in variant_id: 
        img = cv2.GaussianBlur(img, (5, 5), 2)
        noise = np.random.normal(0, 20, img.shape).astype(np.uint8)
        return cv2.add(img, noise)
    elif "Var_10" in variant_id: 
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 10]
        result, encimg = cv2.imencode('.jpg', img, encode_param)
        return cv2.imdecode(encimg, 1)
    return img
    
json_file = 'multi_disease_reports.json'
if not os.path.exists(json_file):
    print(f"错误：找不到文件 {json_file}，请确认数据文件已生成。")
    dummy_data = [{
        "variant_id": "Var_01",
        "report_metadata": {
            "institution": "测试中心医院",
            "report_type": "生化检验报告单",
            "patient_info": {"name": "张三", "sex": "男", "age": "35", "ward": "内科", "clinical_id": "10086"},
            "clinical_diagnosis": "上呼吸道感染",
            "test_date": "2026-03-03"
        },
        "test_results": [
            {"name": "葡萄糖", "abbreviation": "GLU", "value": "5.6", "range": "3.9 ~ 6.1", "unit": "mmol/L"},
            {"name": "钾", "abbreviation": "K", "value": "4.2", "range": "3.5 ~ 5.5", "unit": "mmol/L"}
        ]
    }]
    reports = dummy_data
    print("使用生成的模拟数据继续运行...")
else:
    with open(json_file, 'r', encoding='utf-8') as f:
        reports = json.load(f)

for report in reports:
    try:
        base = create_base_report(report)
        aug_img = apply_augmentation(base, report['variant_id'])
        file_name = f"{report['variant_id']}.png"
        cv2.imwrite(os.path.join(output_dir, file_name), aug_img)
        print(f"已生成 {file_name}")
    except Exception as e:
        print(f"生成 {report.get('variant_id', 'Unknown')} 时出错: {e}")
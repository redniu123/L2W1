import os
import struct
import zipfile
import glob
from PIL import Image
from tqdm import tqdm
import io

def parse_gnt_stream(gnt_stream, output_root, char_counters, filename_hint):
    """
    从内存流中解析 GNT 数据，而不是从物理文件路径解析
    """
    # 获取流的大致大小（用于防止无限读取，虽然流式读取通常依靠内容判断）
    # 这里的 gnt_stream 是 zipfile.open 返回的对象
    
    while True:
        # --- 1. 读取 Sample Size (4 bytes) ---
        header_size_data = gnt_stream.read(4)
        if len(header_size_data) < 4:
            break

        try:
            # little-endian, unsigned int
            sample_size = struct.unpack('<I', header_size_data)[0]
        except struct.error:
            break

        # --- 2. 读取 Tag Code (2 bytes, GBK) ---
        tag_code_data = gnt_stream.read(2)
        if len(tag_code_data) < 2:
            break
            
        try:
            tag_code = tag_code_data.replace(b'\x00', b'')
            label = tag_code.decode('gbk')
        except UnicodeDecodeError:
            label = 'UNK'

        # --- 3. 读取 Width 和 Height (各 2 bytes) ---
        dims_data = gnt_stream.read(4)
        if len(dims_data) < 4:
            break
        
        width = struct.unpack('<H', dims_data[0:2])[0]
        height = struct.unpack('<H', dims_data[2:4])[0]

        # --- 4. 异常检查 (跳过空图) ---
        if width == 0 or height == 0:
            # 跳过剩余数据 (sample_size - 10 header bytes)
            remaining = sample_size - 10
            if remaining > 0: gnt_stream.read(remaining)
            continue

        # --- 5. 读取图像位图 ---
        image_len = width * height
        
        # 安全检查：防止异常数据导致内存溢出
        if image_len > 1024 * 1024 * 5: # 单张图超过5MB视为异常
             # 跳过
            remaining = sample_size - 10
            if remaining > 0: gnt_stream.read(remaining)
            continue

        bitmap = gnt_stream.read(image_len)
        if len(bitmap) != image_len:
            break

        # --- 6. 生成并保存图片 ---
        try:
            image = Image.frombytes('L', (width, height), bitmap)
            
            # 文件名清洗
            safe_label = label
            invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
            for ch in invalid_chars:
                if ch in safe_label:
                    safe_label = safe_label.replace(ch, '_')

            # 全局计数
            if safe_label not in char_counters:
                char_counters[safe_label] = 0
            char_counters[safe_label] += 1
            
            # 生成文件名：真实标签_序号.jpg
            filename = f"{safe_label}_{char_counters[safe_label]:05d}.jpg"
            save_path = os.path.join(output_root, filename)
            
            image.save(save_path)

        except Exception:
            continue

def process_zip_files(zip_list, output_dir):
    """
    处理压缩包列表：直接读取 -> 解析 -> 保存图片
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出目录: {output_dir}")

    # 全局计数器，确保跨压缩包的同名汉字序号连续
    global_char_counters = {}

    for zip_path in zip_list:
        if not os.path.exists(zip_path):
            print(f"⚠️ 文件不存在，跳过: {zip_path}")
            continue

        print(f"\n正在读取压缩包: {zip_path}")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                # 筛选出所有的 .gnt 文件
                gnt_files_in_zip = [f for f in z.namelist() if f.lower().endswith('.gnt')]
                
                print(f" -> 发现 {len(gnt_files_in_zip)} 个 GNT 文件，开始流式解析...")
                
                # 使用 tqdm 显示进度
                for gnt_filename in tqdm(gnt_files_in_zip, desc=f"解析 {os.path.basename(zip_path)}"):
                    # 直接打开压缩包内的文件流
                    with z.open(gnt_filename) as gnt_stream:
                        parse_gnt_stream(gnt_stream, output_dir, global_char_counters, gnt_filename)
                        
        except zipfile.BadZipFile:
            print(f"❌ 错误: {zip_path} 不是有效的 zip 文件。")
        except Exception as e:
            print(f"❌ 处理 {zip_path} 时发生未知错误: {e}")

    print("\n" + "="*30)
    print(f"全部处理完成！")
    print(f"所有图片已保存在: {output_dir}")
    print(f"共提取字符类别数: {len(global_char_counters)}")

if __name__ == "__main__":
    # --- 自动定位当前脚本所在的文件夹 ---
    # 获取当前脚本文件的绝对路径目录
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 打印一下看看路径对不对
    print(f"当前脚本目录: {current_script_dir}")

    # 输入的压缩包列表 (自动拼接完整路径)
    zip_files = [
        os.path.join(current_script_dir, 'Gnt1.0Test.zip'),
        os.path.join(current_script_dir, 'Gnt1.1Test.zip'),
        os.path.join(current_script_dir, 'Gnt1.2Test.zip')
    ]

    # 输出目录也放在当前脚本目录下
    output_folder = os.path.join(current_script_dir, 'casia_sample')

    # 执行
    process_zip_files(zip_files, output_folder)
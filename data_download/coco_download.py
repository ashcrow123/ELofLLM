import os
import requests
import zipfile
from tqdm import tqdm

def download_with_progress(url, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    filename = url.split("/")[-1]
    file_path = os.path.join(save_dir, filename)

    # 若文件已存在，跳过下载
    if os.path.exists(file_path):
        print(f"[✓] 文件已存在：{filename}")
        return file_path

    print(f"↓ 正在下载：{filename}")
    
    # 获取文件大小
    response = requests.head(url)
    file_size = int(response.headers.get('content-length', 0))

    # 开始下载
    with requests.get(url, stream=True) as r, open(file_path, "wb") as f, tqdm(
        total=file_size, unit='B', unit_scale=True, unit_divisor=1024,
        desc=filename, ncols=80
    ) as progress_bar:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                progress_bar.update(len(chunk))

    return file_path

def extract_zip(zip_path, extract_to):
    print(f"↪ 解压中：{os.path.basename(zip_path)}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"[✓] 解压完成：{os.path.basename(zip_path)}")

# 示例：下载 COCO 图像字幕训练集
save_directory = "../coco"

urls = [
    "http://images.cocodataset.org/zips/train2017.zip",
    "http://images.cocodataset.org/zips/val2017.zip",
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
]

for url in urls:
    zip_file_path = download_with_progress(url, save_directory)
    extract_zip(zip_file_path, save_directory)
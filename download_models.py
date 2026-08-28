import os
import sys
import ssl
import urllib.request

ssl._create_default_https_context = ssl._create_unverified_context

# 1. EasyOCR Models
easyocr_dir = os.path.expanduser("~/.EasyOCR/model")
os.makedirs(easyocr_dir, exist_ok=True)

easyocr_models = {
    "craft_mlt_25k.pth": "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/craft_mlt_25k.zip",
    "english_g2.pth": "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip",
}

for name, url in easyocr_models.items():
    target_pth = os.path.join(easyocr_dir, name)
    if not os.path.exists(target_pth):
        zip_path = os.path.join(easyocr_dir, f"{name}.zip")
        print(f"Downloading {name} from {url}...")
        urllib.request.urlretrieve(url, zip_path)
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(easyocr_dir)
        if os.path.exists(zip_path):
            os.remove(zip_path)
        print(f"Extracted {name} successfully.")
    else:
        print(f"EasyOCR model {name} already exists.")

# 2. YOLOX Pretrained Models for Multi-Class Tracking
pretrained_dir = os.path.join(os.path.dirname(__file__), "pretrained")
os.makedirs(pretrained_dir, exist_ok=True)

yolox_models = {
    "yolox_s.pth": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth",
    "yolox_x.pth": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth",
}

for name, url in yolox_models.items():
    target_pth = os.path.join(pretrained_dir, name)
    if not os.path.exists(target_pth):
        print(f"Downloading {name} (multi-class vehicle+human detector) from {url}...")
        urllib.request.urlretrieve(url, target_pth)
        print(f"Downloaded {name} successfully to {target_pth}.")
    else:
        print(f"Pretrained weight {name} already exists.")

print("All models ready!")


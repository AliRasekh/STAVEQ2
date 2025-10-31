import os
import json
import time
import random
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


QA_JSON_PATH = "webvid_1M.json"
SAVE_FOLDER = "webvid/data_mp4"
NUM_WORKERS = 16
MAX_RETRIES = 3

os.makedirs(SAVE_FOLDER, exist_ok=True)


with open(QA_JSON_PATH, "r", encoding="utf-8") as f:
    qa_data = json.load(f)


download_list = []
for item in qa_data:
    videoid = item["videoid"]
    url = item.get("contentUrl") or item.get("content_url")
    if not url:
        continue
    output_path = os.path.join(SAVE_FOLDER, f"{videoid}.mp4")
    if not os.path.exists(output_path):
        download_list.append((url, output_path))

print(f"Found {len(download_list)} videos to download.")

def download_video(url, save_path):
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, stream=True, timeout=15)
            if response.status_code == 200:
                with open(save_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                return True
            else:
                raise Exception(f"HTTP {response.status_code}")
        except Exception as e:
            time.sleep(1 + random.random() * 2)
            if attempt == MAX_RETRIES - 1:
                print(f"Failed to download {url} → {e}")
                return False
    return False


with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = [executor.submit(download_video, url, path) for url, path in download_list]

    for f in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
        pass

print("All downloads complete.")

import os
import csv
import json
from pathlib import Path
from tqdm import tqdm

def collect(csv_dir, max_samples=1_000_000):
    entries = []
    csv_files = sorted(Path(csv_dir).glob("*.csv"))

    for csv_file in tqdm(csv_files, desc="Reading CSV files"):
        with open(csv_file, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                name_len = len(row['name'])
                entries.append({
                    "videoid": row["videoid"],
                    "contentUrl": row["contentUrl"],
                    "duration": row["duration"],
                    "page_dir": row["page_dir"],
                    "name": row["name"],
                    "name_length": name_len
                })


    entries.sort(key=lambda x: x["name_length"], reverse=True)

    return entries[:max_samples]

if __name__ == "__main__":
    csv_folder = "webvid-10M/data/train"
    output_file = "webvid_1M.json"

    results = collect(csv_folder)
    for entry in results:
        del entry["name_length"]

    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(results)} entries with the longest names to {output_file}")

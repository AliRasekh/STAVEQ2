import os
from moviepy.editor import VideoFileClip
from tqdm import tqdm

input_folder = '20bn-something-something-v2'
output_folder = 'data_mp4'

os.makedirs(output_folder, exist_ok=True)

total_files_converted = 0

for filename in tqdm(os.listdir(input_folder)):

    if filename.endswith('.webm'):
        input_path = os.path.join(input_folder, filename)
        output_filename = f"{os.path.splitext(filename)[0]}.mp4"
        output_path = os.path.join(output_folder, output_filename)

        if not os.path.exists(output_path):
            total_files_converted += 1
            with VideoFileClip(input_path) as video:
                video.write_videofile(output_path, codec='libx264',
                                      verbose=False, logger=None)


print("\ncompleted!")
print(f"{total_files_converted} files converted.")

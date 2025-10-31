First, run ssv2.sh in data/ssv2 folder to download Something-Something-v2 dataset, then run convert.py script to convert them to mp4.

Then download Webvid-10M dataset partitions from https://huggingface.co/datasets/TempoFunk/webvid-10M

After downloading, use data/Webvid_1M.py to extract around 1 milion samples, ensuring that you have as much data as you want for training.

Then use data/download_webvid.py to download the videos for Webvid.


For each experiment, run select_subset.py before anything to create the subset for the experiment.


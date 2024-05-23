# -- coding: utf-8 --**

import json
import os

json_path = 'fashion_dataset.json'
video_dir = '/work00/magic_animate_unofficial/data/fashion_dataset'
contents = []
for video in os.listdir(video_dir):
    content = os.path.join(video_dir, video)
    contents.append(content)

with open(json_path, 'w') as f:
    json.dump(contents, f)

dataset = json.load(open(json_path))
print(dataset)

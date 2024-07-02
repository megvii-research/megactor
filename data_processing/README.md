## Dataset

You may refer to a 10-min dataset in the following format at [**Google Drive**](https://drive.google.com/drive/folders/1GVhCd3syxl2-oqF7TiPyoy7VrWJXbrQs?usp=drive_link).

For better data organization and ease of storage and retrieval, we have bundled video clips into tar archives. The specific format is shown below.

```
Training Dataset:
|----train_set
    |----VFHQ
        |----group124.tar
        |----group125.tar
        |----group126.tar
        ...
    |----CelebV
        |----archive0_1_560.tar
        |----archive0_1_561.tar
        |----archive0_1_562.tar
        ...
```

We utilize the webdataset framework to package video data into corresponding tar archives. For each tar archive:

```
For example xxx.tar
|----video_seq1: (dict)
    |----mp4: (byte) # video clips NxHxWxC
    |----swapped.mp4: (byte)  # swapped video clips NxHxWxC
    |----mp4_styled: (byte) # styled video clips NxHxWxC
    |----dwpose_result.pyd: (Optional) (byte) # landmark
    |----dwpose_score.pyd: (Optional) (byte) # landmark score
    |----faces_bbox.pyd": (Optional) (byte) # face bounding box
|----video_seq2: (dict)
    |----mp4: (byte) # video clips NxHxWxC
    |----swapped.mp4: (byte)  # swapped video clips NxHxWxC
    |----mp4_styled: (byte) # styled video clips NxHxWxC
    |----dwpose_result.pyd: (Optional) (byte) # landmark
    |----dwpose_score.pyd: (Optional) (byte) # landmark score
    |----faces_bbox.pyd": (Optional) (byte) # face bounding box
...
```

### 1.0 Data Packing


We demonstrate the process of packaging and generating synthetic data from a subset of the VFHQ dataset as an example for handling Raw data.

```
cd data_processing && ln -s ../RawVideoDriven/weights weights
```


```
python3 1.pack_data.py --zipfile ./VFHQ/group124.zip --dstfile ./VFHQ/group124.tar
```


### 2.0 Generating DWPose


We utilize dwpose to generate facial landmarks and face bounding boxes, which are employed for subsequent data curation.


```
python3 2.generate_dwpose.py --tarfile ./VFHQ/group124.tar --dstfile ./VFHQ/group124_dwpose.tar
```


### 3.0 Synthesizing stylized data

```
python3 3.style_transfer.py --tarfile ./VFHQ/group124_dwpose.tar --dstfile ./VFHQ/group124_dwpose_style.tar
```

### 4.0 Synthesizing face-swapping data

The face-swapping example code is provided below, with the requirement to install [FaceFusion](https://github.com/facefusion/facefusion)

```
git clone https://github.com/facefusion/facefusion
cd facefusion
python install.py
python run.py
```

```
python3 4.0 face_fusion.py --tar ./VFHQ/group124_dwpose_style.tar --dstfile ./VFHQ/group124_dwpose_style_facefusion.tar --source_images your_source_images_dir 
```

### 5.0 Eye-tracking data filtering strategy.

Eye-tracking data curation requires the installation of the  [L2CS-Net](https://github.com/Ahmednull/L2CS-Net)

```
git clone https://github.com/Ahmednull/L2CS-Net
cd L2CS-Net
pip install -e .
Download [L2CS-Net Checkpoint](https://drive.google.com/drive/folders/1qDzyzXO6iaYIMDJDSyfKeqBx8O74mF8s) to L2CS-Net/models/L2CSNet_gaze360.pkl
```


```
python3 5.filter_eye.py --tarfile ./VFHQ/group124_dwpose_style_facefusion.tar --dstfile ./VFHQ/group124_dwpose_style_facefusion_eye-filter.tar
```





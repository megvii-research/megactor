import webdataset as wds
from webdataset import gopen, gopen_schemes
from tqdm import tqdm 
import imageio.v3 as iio
import numpy as np
import traceback
from tempfile import NamedTemporaryFile
import subprocess
import random


def worker(job):
    src_tarfilepath, dst_tarfilepath, SOURCE_IMAGE_PATH_LIST = job
    print(f"Processing {src_tarfilepath} -> {dst_tarfilepath}")
    
    err_msg = None
    try:
        # 如果字符串中有()的话，必须添加转义符号\
        src_tarfilepath = src_tarfilepath.replace("(","\(")
        src_tarfilepath = src_tarfilepath.replace(")","\)")

        dst_tarfilepath = dst_tarfilepath.replace("(","\(")
        dst_tarfilepath = dst_tarfilepath.replace(")","\)")

        dataset = wds.WebDataset(src_tarfilepath)

        with open(dst_tarfilepath, "wb") as wds_wf:
            sink = wds.TarWriter(fileobj=wds_wf,)
            for data in tqdm(dataset):
                # key          = data["__key__"]
                # url          = data["__url__"]
                video_bytes  = data["mp4"]
                swap_video_bytes = None
                target_tempfile = NamedTemporaryFile(delete=True, suffix='.mp4')
                output_tempfile =  NamedTemporaryFile(delete=True, suffix='.mp4')
                random_index = random.randint(0, len(SOURCE_IMAGE_PATH_LIST))
                source_image_path = SOURCE_IMAGE_PATH_LIST[random_index]
                try:
                    with open(target_tempfile.name, mode="wb") as wf:
                        wf.write(video_bytes)
                    
                    command = [
                        "python3",
                        "./facefusion/run.py",
                        "--headless",
                        "--skip-download",
                        "--execution-providers", "cuda",
                        "--execution-thread-count", "1",
                        "--source", source_image_path,
                        "--target", f"{target_tempfile.name}",#"/data/tempfile/DFEW/part_1/886.mp4",
                        "--output", f"{output_tempfile.name}",
                    ]
                    subprocess.run(command, check=True)
                
                    with open(output_tempfile.name, "rb") as rf:
                        swap_video_bytes = rf.read()
    
                    data["swapped.mp4"] = swap_video_bytes
                    data["source.png"] = iio.imread(source_image_path)
                    data["source_path"] = source_image_path
                    sink.write(data)
                except Exception as e:
                    traceback.print_exc()
                finally:
                    target_tempfile.close()
                    output_tempfile.close()
                
            sink.close()
        dataset.close()
    except Exception as e:
        print(e)
        err_msg = e
    return src_tarfilepath, err_msg


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tarfile', type=str, default="")
    parser.add_argument('--dstfile', type=str, default="")
    parser.add_argument('--source_images', type=str, default="")
    args = parser.parse_args()    
    
    SOURCE_IMAGE_PATH_LIST = glob.glob(f"{args.source_images}/*.png")
    out_dir = args.out_dir

    jobs = [(args.tarfile, args.dstfile, SOURCE_IMAGE_PATH_LIST)]
    
    for job in tqdm(jobs):
        src_tarfilepath, err_msg = worker(job)
        print(src_tarfilepath, err_msg)
        
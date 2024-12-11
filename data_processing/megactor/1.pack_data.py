import refile
import webdataset as wds
import zipfile
import itertools
import numpy as np
import imageio.v3 as iio
from tqdm import tqdm


def worker(job):
    src_tarfilepath, dst_tarfilepath = job
    err_msg = None
    try:
        assert src_tarfilepath.endswith(".zip"), "VFHQ应该都是以.zip格式存储的..."
        with open(src_tarfilepath, "rb") as rf: 
            zipobj = zipfile.ZipFile(rf,mode="r")
            filename_list = zipobj.namelist()
            with open(dst_tarfilepath, "wb") as wf:
                sink = wds.TarWriter(fileobj=wf,)
                for video_key, framename_list in tqdm(itertools.groupby(filename_list, key=lambda x: x.split("/")[0])):
                    frames = []
                    framename_list = sorted(list(framename_list))
                    for framename in framename_list:
                        frame_bytes = zipobj.read(framename)
                        frames.append(iio.imread(frame_bytes))
                    frames = np.array(frames)
                    n,h,w,c = frames.shape  #h264要求H/W必须是偶数
                    frames = frames[:,:(h//2)*2, :(w//2)*2, :]
                    video_bytes = iio.imwrite("<bytes>", frames, extension='.mp4', plugin="pyav", codec="h264", fps=30)
                
                    sink.write({
                            "__key__": video_key,
                            # "__url__": url,
                            "mp4"    : video_bytes,
                        })
                    del frames
                    del video_bytes
                sink.close()
            zipobj.close()
    except Exception as e:
        print(e)
        err_msg = e
    return src_tarfilepath, err_msg


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--zipfile', type=str, default="")
    parser.add_argument('--dstfile', type=str, default="")
    args = parser.parse_args()    
    

    jobs = [
        (args.zipfile, args.dstfile),
    ]

    for job in tqdm(jobs):
        src_tarfilepath, err_msg = worker(job)

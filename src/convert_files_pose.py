# Converts all the files in a given directory into csv and rendered videos.
# This is a workaround while file iteration with mediapipe breaks because of egl exception on Linux.

import os
import sys
import multiprocessing as mp
from datetime import datetime
def start():
    startTime = datetime.now()
    pool = mp.Pool(mp.cpu_count())
    os.system("bazel build -c opt --copt -DMESA_EGL_NO_X11_HEADERS app:pose_to_csv_gpu")
    input_dir = "/home/signlang/Videos/input/"

    filenames = []
    for subdir, dirs, files in os.walk(input_dir):
        for file in files:
            filenames.append(os.path.join(subdir, file))
    
    pool.map_async(doConversion, [file for file in filenames]).get() # 0:16:27.888996
    endTime = datetime.now()
    print("Duration:", endTime - startTime)
            

def doConversion(filename):
    result = os.system("GLOG_logtostderr=0 bazel-bin/app/pose_to_csv_gpu \
    --calculator_graph_config_file=graphs/pose_to_csv_gpu.pbtxt \
    --input_video_path="+ filename + " \
    --output_video_path=/home/signlang/Videos/output_pose/")
    print (filename, result)

if __name__ == "__main__":
    start()
# Converts all the files in a given directory into csv and rendered videos.
# This is a workaround while file iteration with mediapipe breaks because of egl exception on Linux.

import os
import sys
import multiprocessing as mp
def start():
    pool = mp.Pool(mp.cpu_count())
    os.system("cd src/ &&  bazel build -c opt --copt -DMESA_EGL_NO_X11_HEADERS app:files_to_csv_gpu")
    input_dir = "/home/signlang/Videos/input/"

    filenames = []
    for subdir, dirs, files in os.walk(input_dir):
        for file in files:
            filenames.append(os.path.join(subdir, file))
    pool.map_async(doConversion, [file for file in filenames]).get()
    # for subdir, dirs, files in os.walk(input_dir):
    #     for file in files:
    #         filename = os.path.join(subdir, file)
    #         result = os.system("cd src/ && GLOG_logtostderr=0 bazel-bin/app/files_to_csv_gpu \
    #         --calculator_graph_config_file=graphs/video_processing_gpu.pbtxt \
    #         --input_video_path="+ filename + " \
    #         --output_video_path=/home/signlang/Videos/output_relative_2D/")
    #         print(subdir, file, result)
            

def doConversion(filename):
    result = os.system("cd src/ && GLOG_logtostderr=0 bazel-bin/app/files_to_csv_gpu \
    --calculator_graph_config_file=graphs/video_processing_gpu.pbtxt \
    --input_video_path="+ filename + " \
    --output_video_path=/home/signlang/Videos/output_relative_2D/")
    print (filename, result)

if __name__ == "__main__":
    start()
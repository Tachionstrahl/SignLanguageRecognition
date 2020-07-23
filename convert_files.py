# Converts all the files in a given directory into csv and rendered videos.
# This is a workaround while file iteration with mediapipe breaks because of egl exception on Linux.

import os
import sys
def start():
    os.system("cd src/ &&  bazel build -c opt --copt -DMESA_EGL_NO_X11_HEADERS app:files_to_csv_gpu")
    input_dir = "/home/datagroup/Videos/signlang/"
    for subdir, dirs, files in os.walk(input_dir):
        for file in files:
            filename = os.path.join(subdir, file)
            result = os.system("cd src/ && GLOG_logtostderr=1 bazel-bin/app/files_to_csv_gpu \
            --calculator_graph_config_file=graphs/video_processing_gpu.pbtxt \
            --input_video_path="+ filename + " \
            --output_video_path=/home/datagroup/Videos/rendered3D")
            

if __name__ == "__main__":
    start()
# Converts all the files in a given directory into csv and rendered videos.
# This is a workaround while file iteration with mediapipe breaks because of egl exception on Linux.

import os
import sys
def start():
    os.system("cd src/ &&  bazel build -c opt --copt -DMESA_EGL_NO_X11_HEADERS app:files_to_csv_gpu")
    input_dir = "/home/michi/Videos/4_zu_3/"
    for subdir, dirs, files in os.walk(input_dir):
        for file in files:
            filename = os.path.join(subdir, file)
            result = os.system("cd src/ && GLOG_logtostderr=0 bazel-bin/app/files_to_csv_gpu \
            --calculator_graph_config_file=graphs/video_processing_gpu.pbtxt \
            --input_video_path="+ filename + " \
            --output_video_path=/home/michi/Videos/4_zu_3_rendered_2D/")
            print(subdir, file, result)
            

if __name__ == "__main__":
    start()
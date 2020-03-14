bazel build -c opt --copt -DMESA_EGL_NO_X11_HEADERS app:files_to_csv_gpu && GLOG_logtostderr=1 bazel-bin/app/files_to_csv_gpu --calculator_graph_config_file=graphs/video_processing_gpu.pbtxt

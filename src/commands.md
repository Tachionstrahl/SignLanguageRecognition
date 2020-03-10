# Command list
### Information
- Follow the installation guide of MediaPipe first.
- All commands are relative to this folder. (`src/`)


## Files to CSV on CPU
### Build the source files
`bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 app:files_to_csv_cpu`
### Run the application
`GLOG_logtostderr=1 bazel-bin/app/files_to_csv_cpu --calculator_graph_config_file=graphs/video_processing_cpu.pbtxt`

## Files to CSV on GPU
### Build the source files
`bazel build -c opt --copt -DMESA_EGL_NO_X11_HEADERS app:files_to_csv_gpu`
### Run the application
`GLOG_logtostderr=1 bazel-bin/app/files_to_csv_gpu --calculator_graph_config_file=graphs/video_processing_gpu.pbtxt`
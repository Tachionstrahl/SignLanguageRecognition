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
#### Options
Processing Videos in a given directory requires two arguments:
1. The input root directory containing subdirectories with .mp4 files. `--input_video_path=/home/Videos/SL/input`
   Example for a structure: 
   ```
   input
    -- word_one
        -- file_one.mp4
        -- file_two.mp4
    -- word_two
        -- file_three.mp4
    -- word_three
        -- file_four.mp4
   ```
2. The ouput root directory. Can be empty. `--output_video_path="/home/Videos/SL/output`
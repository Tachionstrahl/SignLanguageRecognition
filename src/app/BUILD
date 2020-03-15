package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "files_to_csv_cpu",
    deps = [
        "@mediapipe//mediapipe/examples/desktop:demo_run_graph_main",
        "//graphs/subgraphs:detections_cpu",
        "//calculators:detections_to_csv_calculator"
    ],
)

cc_binary(
    name = "files_to_csv_gpu",
    deps = [
        ":run_graph_main_gpu",
        "//graphs/subgraphs:detections_gpu",
        "//calculators:detections_to_csv_calculator"
    ],
)

# Linux only.
# Must have a GPU with EGL support:
# ex: sudo apt-get install mesa-common-dev libegl1-mesa-dev libgles2-mesa-dev
# (or similar nvidia/amd equivalent)
cc_library(
    name = "run_graph_main_gpu",
    srcs = ["run_graph_main_gpu.cc"],
    deps = [
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe//mediapipe/framework/formats:image_frame",
        "@mediapipe//mediapipe/framework/formats:image_frame_opencv",
        "@mediapipe//mediapipe/framework/port:commandlineflags",
        "@mediapipe//mediapipe/framework/port:file_helpers",
        "@mediapipe//mediapipe/framework/port:opencv_highgui",
        "@mediapipe//mediapipe/framework/port:opencv_imgproc",
        "@mediapipe//mediapipe/framework/port:opencv_video",
        "@mediapipe//mediapipe/framework/port:parse_text_proto",
        "@mediapipe//mediapipe/framework/port:status",
        "@mediapipe//mediapipe/gpu:gl_calculator_helper",
        "@mediapipe//mediapipe/gpu:gpu_buffer",
        "@mediapipe//mediapipe/gpu:gpu_shared_data_internal",
    ],
)
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

cc_binary(
    name = "pose_to_csv_gpu",
    deps = [
        ":run_graph_main_gpu",
        "//graphs/subgraphs:pose_tracking_gpu",
        "//calculators:pose_landmarks_to_csv_calculator"
    ],
)

cc_binary(
    name = "prediction_gpu",
    deps = [
        "@mediapipe//mediapipe/examples/desktop:demo_run_graph_main_gpu",
        "@mediapipe//mediapipe/calculators/util:annotation_overlay_calculator",
        "@mediapipe//mediapipe/calculators/core:constant_side_packet_calculator",
        "@mediapipe//mediapipe/calculators/util:local_file_contents_calculator",
        "//calculators:text_to_render_data_calculator",
        "//calculators:sign_lang_prediction_calculator",
        "//graphs/subgraphs:detections_gpu"
    ]
)

cc_binary(
    name = "prediction_cpu",
    deps = [
        "@mediapipe//mediapipe/examples/desktop:demo_run_graph_main",
        "@mediapipe//mediapipe/calculators/util:annotation_overlay_calculator",
        "@mediapipe//mediapipe/calculators/core:constant_side_packet_calculator",
        "@mediapipe//mediapipe/calculators/util:local_file_contents_calculator",
        "//calculators:text_to_render_data_calculator",
        "//calculators:sign_lang_prediction_calculator",
        "//graphs/subgraphs:detections_cpu"
    ]
)

cc_binary(
    name = "pose_prediction_gpu",
    deps = [
        "@mediapipe//mediapipe/examples/desktop:demo_run_graph_main_gpu",
        "@mediapipe//mediapipe/calculators/util:annotation_overlay_calculator",
        "@mediapipe//mediapipe/calculators/core:constant_side_packet_calculator",
        "@mediapipe//mediapipe/calculators/util:local_file_contents_calculator",
        "//calculators:text_to_render_data_calculator",
        "//calculators:sign_lang_prediction_calculator",
        "//graphs/subgraphs:pose_tracking_gpu"
    ]
)

# Linux only.
# Must have a GPU with EGL support:
# ex: sudo apt-get install mesa-common-dev libegl1-mesa-dev libgles2-mesa-dev
# (or similar nvidia/amd equivalent)
cc_library(
    name = "run_graph_main_gpu",
    srcs = ["run_graph_main_gpu.cc"],
    deps = [
        "@boost//:filesystem",
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
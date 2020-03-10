
cc_library(
    name = "detections_to_csv_calculator",
    srcs = ["detections_to_csv_calculator.cc"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe//mediapipe/framework/port:ret_check",
        "@mediapipe//mediapipe/framework/port:status",
        "@mediapipe//mediapipe/framework/formats:landmark_cc_proto",
        "@mediapipe//mediapipe/framework/formats:detection_cc_proto"
    ],
    alwayslink = 1,
)
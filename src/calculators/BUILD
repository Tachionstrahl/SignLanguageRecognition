
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

cc_library(
    name = "text_to_render_data_calculator",
    srcs = ["text_to_render_data_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe//mediapipe/framework/port:status",
        "@mediapipe//mediapipe/util:color_cc_proto",
        "@mediapipe//mediapipe/util:render_data_cc_proto",
    ],
    alwayslink = 1,
)

cc_library(
    name = "sign_language_prediction_calculator",
    srcs = ["sign_language_prediction_calculator.cc"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe//mediapipe/framework/port:ret_check",
        "@mediapipe//mediapipe/framework/port:status",
        "@mediapipe//mediapipe/framework/formats:landmark_cc_proto",
        "@mediapipe//mediapipe/framework/formats:detection_cc_proto",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
        "@org_tensorflow//tensorflow/lite/delegates/xnnpack:xnnpack_delegate"
    ],
    alwayslink = 1,
)
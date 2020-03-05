
package(default_visibility = ["//visibility:public"])

load(
    "//:mediapipe_graph.bzl",
    "mediapipe_binary_graph",
)

cc_library(
    name = "sign_lang_calculators",
    deps = [
        "//calculators:detections_to_csv_calculator",
    ],
)


mediapipe_binary_graph(
    name = "test_graph",
    graph = "test.pbtxt",
    output_name = "test.binarypb",
    deps = [":sign_lang_calculators"]
)
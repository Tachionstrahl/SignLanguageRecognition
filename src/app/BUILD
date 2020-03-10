package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "files_to_csv_cpu",
    deps = [
        "@mediapipe//mediapipe/examples/desktop:demo_run_graph_main",
        "//graphs/subgraphs:detections_desktop_cpu",
        "//calculators:detections_to_csv_calculator"
    ],
)

cc_binary(
    name = "files_to_csv_gpu",
    deps = [
        "@mediapipe//mediapipe/examples/desktop:demo_run_graph_main_gpu",
        "//graphs/subgraphs:detections_gpu",
        "//calculators:detections_to_csv_calculator"
    ],
)
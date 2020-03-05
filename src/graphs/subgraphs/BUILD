load(
    "@//tool:mediapipe_graph.bzl",
    "mediapipe_simple_subgraph",
)

package(default_visibility = ["//visibility:public"])

mediapipe_simple_subgraph(
    name = "detections_desktop_cpu",
    graph = "detections_desktop_cpu.pbtxt",
    register_as = "DetectionsDesktopCpuSubgraph",
    deps = [
        "@mediapipe//mediapipe/examples/desktop:demo_run_graph_main",
        "@mediapipe//mediapipe/graphs/hand_tracking:multi_hand_desktop_tflite_calculators",
        "@mediapipe//mediapipe/graphs/face_detection:desktop_tflite_calculators",
        "@mediapipe//mediapipe/calculators/core:merge_calculator",
        # "@mediapipe//mediapipe/calculators/util:collection_has_min_size_calculator",
        # "@mediapipe//mediapipe/calculators/core:gate_calculator",
        # "@mediapipe//mediapipe/graphs/hand_tracking/subgraphs:multi_hand_detection_cpu",
        # "@mediapipe//mediapipe/graphs/hand_tracking/subgraphs:multi_hand_landmark_cpu",
        # "@mediapipe//mediapipe/calculators/core:previous_loopback_calculator",
        # "@mediapipe//mediapipe/calculators/util:association_norm_rect_calculator",
        # "@mediapipe//mediapipe/graphs/hand_tracking/subgraphs:multi_hand_renderer_gpu",
    ],
)
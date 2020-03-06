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

mediapipe_simple_subgraph(
    name = "detections_gpu",
    graph = "detections_gpu.pbtxt",
    register_as = "DetectionsGPUSubgraph",
    deps = [
        ":face_detection_subgraph",
        "@mediapipe//mediapipe/examples/desktop:demo_run_graph_main",
        "@mediapipe//mediapipe/graphs/hand_tracking:multi_hand_desktop_tflite_calculators",
        "@mediapipe//mediapipe/graphs/face_detection:desktop_tflite_calculators",
        "@mediapipe//mediapipe/calculators/core:merge_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "face_detection_subgraph",
    graph = "face_detection_subgraph.pbtxt",
    register_as = "FaceDetectionSubgraph",
    deps = [        
	    "@mediapipe//mediapipe/calculators/core:flow_limiter_calculator",
        "@mediapipe//mediapipe/calculators/image:image_transformation_calculator",
        "@mediapipe//mediapipe/calculators/tflite:ssd_anchors_calculator",
        "@mediapipe//mediapipe/calculators/tflite:tflite_converter_calculator",
        "@mediapipe//mediapipe/calculators/tflite:tflite_inference_calculator",
        "@mediapipe//mediapipe/calculators/tflite:tflite_tensors_to_detections_calculator",
        "@mediapipe//mediapipe/calculators/util:annotation_overlay_calculator",
        "@mediapipe//mediapipe/calculators/util:detection_label_id_to_text_calculator",
        "@mediapipe//mediapipe/calculators/util:detection_letterbox_removal_calculator",
        "@mediapipe//mediapipe/calculators/util:detections_to_render_data_calculator",
        "@mediapipe//mediapipe/calculators/util:non_max_suppression_calculator",
        "@mediapipe//mediapipe/gpu:gpu_buffer_to_image_frame_calculator",
        "@mediapipe//mediapipe/gpu:image_frame_to_gpu_buffer_calculator",
	],
)
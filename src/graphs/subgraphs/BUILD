load(
    "@//tool:mediapipe_graph.bzl",
    "mediapipe_simple_subgraph",
)

package(default_visibility = ["//visibility:public"])

mediapipe_simple_subgraph(
    name = "detections_cpu",
    graph = "detections_cpu.pbtxt",
    register_as = "DetectionsCPUSubgraph",
    deps = [
        ":multi_hand_detections_to_render_data",
        "@mediapipe//mediapipe/graphs/hand_tracking:multi_hand_desktop_tflite_calculators",
        "@mediapipe//mediapipe/graphs/face_detection:desktop_tflite_calculators",
        "@mediapipe//mediapipe/calculators/core:begin_loop_calculator",
        "@mediapipe//mediapipe/calculators/core:clip_vector_size_calculator",
        "@mediapipe//mediapipe/calculators/core:end_loop_calculator",
        "@mediapipe//mediapipe/calculators/image:image_properties_calculator",
        "@mediapipe//mediapipe/calculators/image:image_transformation_calculator",
        "@mediapipe//mediapipe/calculators/tflite:ssd_anchors_calculator",
        "@mediapipe//mediapipe/calculators/tflite:tflite_converter_calculator",
        "@mediapipe//mediapipe/calculators/tflite:tflite_custom_op_resolver_calculator",
        "@mediapipe//mediapipe/calculators/tflite:tflite_inference_calculator",
        "@mediapipe//mediapipe/calculators/tflite:tflite_tensors_to_detections_calculator",
        "@mediapipe//mediapipe/calculators/util:detection_label_id_to_text_calculator",
        "@mediapipe//mediapipe/calculators/util:detection_letterbox_removal_calculator",
        "@mediapipe//mediapipe/calculators/util:detections_to_rects_calculator",
        "@mediapipe//mediapipe/calculators/util:non_max_suppression_calculator",
        "@mediapipe//mediapipe/calculators/util:rect_transformation_calculator"
    ],
)

mediapipe_simple_subgraph(
    name = "multi_hand_detections_to_render_data",
    graph = "multi_hand_detections_to_render_data_subgraph.pbtxt",
    register_as = "MultiHandDetectionsToRenderDataSubgraph",
    deps = [
        "@mediapipe//mediapipe/calculators/core:begin_loop_calculator",
        "@mediapipe//mediapipe/calculators/core:end_loop_calculator",
        "@mediapipe//mediapipe/calculators/util:annotation_overlay_calculator",
        "@mediapipe//mediapipe/calculators/util:detections_to_render_data_calculator",
        "@mediapipe//mediapipe/calculators/util:landmarks_to_render_data_calculator",
        "@mediapipe//mediapipe/calculators/util:rect_to_render_data_calculator",
    ]
)

mediapipe_simple_subgraph(
    name = "detections_gpu",
    graph = "detections_gpu.pbtxt",
    register_as = "DetectionsGPUSubgraph",
    deps = [
        ":multi_hand_detections_to_render_data",
        "@mediapipe//mediapipe/graphs/hand_tracking:multi_hand_mobile_calculators",
        "@mediapipe//mediapipe/graphs/face_detection:mobile_calculators"
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

mediapipe_simple_subgraph(
    name = "hand_and_face_renderer_gpu",
    graph = "hand_and_face_renderer_gpu.pbtxt",
    register_as = "HandAndFaceRendererSubgraph",
    deps = [
        "@mediapipe//mediapipe/calculators/core:begin_loop_calculator",
        "@mediapipe//mediapipe/calculators/core:end_loop_calculator",
        "@mediapipe//mediapipe/calculators/util:annotation_overlay_calculator",
        "@mediapipe//mediapipe/calculators/util:detections_to_render_data_calculator",
        "@mediapipe//mediapipe/calculators/util:landmarks_to_render_data_calculator",
        "@mediapipe//mediapipe/calculators/util:rect_to_render_data_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "pose_tracking_gpu",
    graph = "pose_tracking_gpu.pbtxt",
    register_as = "PoseTrackingGpuSubgraph",
    deps = [
        "@mediapipe//mediapipe/graphs/pose_tracking:upper_body_pose_tracking_gpu_deps"
    ]
)
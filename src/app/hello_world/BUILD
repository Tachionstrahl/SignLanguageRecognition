
package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "hello_world",
    srcs = ["hello_world.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "@mediapipe//mediapipe/calculators/core:pass_through_calculator",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe//mediapipe/framework/port:parse_text_proto"
    ],
)

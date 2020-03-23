
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_builder.h"
#include "mediapipe/util/render_data.pb.h"

namespace mediapipe {

// Takes in a std::string, draws the text std::string by cv::putText(), and
// outputs an ImageFrame.
//
// Example config:
// node {
//   calculator: "TextToRenderDataCalculator"
//   input_stream: "text_to_put"
//   input_stream: "image_frame"
//   output_stream: "out_image_frames"
// }
// TODO: Generalize the calculator for other text use cases.
class TextToRenderDataCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc);
  ::mediapipe::Status Process(CalculatorContext* cc) override;
};

::mediapipe::Status TextToRenderDataCalculator::GetContract(CalculatorContract* cc) {
  cc->Inputs().Index(0).Set<std::string>();
  cc->Outputs().Index(0).Set<mediapipe::RenderData>();
  return ::mediapipe::OkStatus();
}

::mediapipe::Status TextToRenderDataCalculator::Process(CalculatorContext* cc) {
  const std::string& text_content = cc->Inputs().Index(0).Get<std::string>();
  cv::Mat mat = cv::Mat::zeros(640, 640, CV_8UC4);
  cv::putText(mat, text_content, cv::Point(15, 70), cv::FONT_HERSHEY_PLAIN, 3,
              cv::Scalar(255, 255, 0, 255), 4);
  std::unique_ptr<ImageFrame> output_frame = absl::make_unique<ImageFrame>(
      ImageFormat::SRGBA, mat.size().width, mat.size().height);
  mat.copyTo(formats::MatView(output_frame.get()));
  cc->Outputs().Index(0).Add(output_frame.release(), cc->InputTimestamp());
  return ::mediapipe::OkStatus();
}

REGISTER_CALCULATOR(TextToRenderDataCalculator);

}  // namespace mediapipe

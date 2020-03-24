
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/render_data.pb.h"

using namespace mediapipe;

namespace signlang {

// Takes in a std::string, draws the text std::string by cv::putText(), and
// outputs an ImageFrame.
//
// Example config:
// node {
//   calculator: "TextToRenderDataCalculator"
//   input_stream: "text_to_put"
//   output_stream: "render_data"
// }
class TextToRenderDataCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc);
  ::mediapipe::Status Process(CalculatorContext* cc) override;
};

::mediapipe::Status TextToRenderDataCalculator::GetContract(CalculatorContract* cc) {
  cc->Inputs().Index(0).SetAny();
  cc->Outputs().Index(0).Set<mediapipe::RenderData>();
  return ::mediapipe::OkStatus();
}

::mediapipe::Status TextToRenderDataCalculator::Process(CalculatorContext* cc) {
  // As an example, please see also mediapipe/calculators/util/labels_to_render_data_calculator.cc
  //const std::string& text_content = cc->Inputs().Index(0).Get<std::string>();
  // For Debug purposes:
  const std::string& text_content = "Test";
  RenderData render_data;
  auto* text_annotation = render_data.add_render_annotations();
  text_annotation->set_thickness(2);
  text_annotation->mutable_color()->set_r(255);
  text_annotation->mutable_color()->set_g(0);
  text_annotation->mutable_color()->set_b(0);
 
  auto* text = text_annotation->mutable_text();
  text->set_display_text(text_content);
  text->set_font_height(30.0);
  text->set_left(300.0);
  text->set_baseline(300.0);
  text->set_font_face(2);
  cc->Outputs()
      .Index(0)
      .AddPacket(mediapipe::MakePacket<RenderData>(render_data).At(cc->InputTimestamp()));
  return ::mediapipe::OkStatus();
}

REGISTER_CALCULATOR(TextToRenderDataCalculator);

}  // namespace mediapipe

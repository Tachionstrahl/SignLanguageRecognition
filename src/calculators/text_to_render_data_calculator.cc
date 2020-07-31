
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

 private:
  double video_height = 0.0;
  double font_height = 30.0;
  double text_margin = 10.0;
};

::mediapipe::Status TextToRenderDataCalculator::GetContract(CalculatorContract* cc) {
  cc->Inputs().Index(0).Set<std::string>();
  cc->Outputs().Index(0).Set<mediapipe::RenderData>();
  return ::mediapipe::OkStatus();
}

::mediapipe::Status TextToRenderDataCalculator::Process(CalculatorContext* cc) {
  // As an example, please see also mediapipe/calculators/util/labels_to_render_data_calculator.cc
  auto& text_content = cc->Inputs().Index(0).Get<std::string>();

  RenderData render_data;

  auto* text_annotation = render_data.add_render_annotations();
  text_annotation->set_thickness(2.0);
  text_annotation->mutable_color()->set_r(0);
  text_annotation->mutable_color()->set_g(0);
  text_annotation->mutable_color()->set_b(0);
 
  auto* text = text_annotation->mutable_text();
  text->set_display_text(text_content);
  text->set_font_height(font_height);
  text->set_left(text_margin);
  text->set_baseline(video_height + font_height + text_margin);
  // text->set_font_face(0);

  cc->Outputs()
      .Index(0)
      .AddPacket(mediapipe::MakePacket<RenderData>(render_data).At(cc->InputTimestamp()));

  return ::mediapipe::OkStatus();
}

REGISTER_CALCULATOR(TextToRenderDataCalculator);

}  // namespace signlang

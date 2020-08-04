
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/render_data.pb.h"

using namespace mediapipe;

namespace signlang {

constexpr char kTextTag[] = "TEXT";
constexpr char kSizeTag[] = "SIZE";
constexpr char kRenderDataTag[] = "RENDER_DATA";

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
  int image_width;
  int image_height;
  int box_width = 400;
  int box_height = 50;
  int box_margin_bottom = 30;
  double font_height = 15;
  double text_margin = 10.0;
};

::mediapipe::Status TextToRenderDataCalculator::GetContract(CalculatorContract* cc) {
  cc->Inputs().Tag(kTextTag).Set<std::string>();
  cc->Inputs().Tag(kSizeTag).Set<std::pair<int,int>>();
  cc->Outputs().Get(kRenderDataTag, 0).Set<mediapipe::RenderData>();
  cc->Outputs().Get(kRenderDataTag, 1).Set<mediapipe::RenderData>();
  return ::mediapipe::OkStatus();
}

::mediapipe::Status TextToRenderDataCalculator::Process(CalculatorContext* cc) {
  // As an example, please see also mediapipe/calculators/util/labels_to_render_data_calculator.cc
  auto& text_content = cc->Inputs().Tag(kTextTag).Get<std::string>();
  absl::optional<std::pair<int, int>> image_size = cc->Inputs().Tag(kSizeTag).Get<std::pair<int,int>>();
  image_width = image_size->first;
  image_height = image_size->second;

  RenderData render_data;
  int margin_left_right = (image_width - box_width) / 2;
  auto* text_annotation = render_data.add_render_annotations();
  text_annotation->set_thickness(1.5);
  text_annotation->mutable_color()->set_r(255);
  text_annotation->mutable_color()->set_g(255);
  text_annotation->mutable_color()->set_b(255);
 
  auto* text = text_annotation->mutable_text();
  text->set_display_text(text_content);
  text->set_font_height(font_height);
  text->set_left(margin_left_right + text_margin);
  text->set_baseline(image_height - box_margin_bottom - ((box_height - font_height) / 2));
  text->set_font_face(2);

  // Black background for text
  RenderData background_render_data;
  auto* annotation = background_render_data.add_render_annotations();
  auto* rectangle = annotation->mutable_filled_rectangle()->mutable_rectangle();
  annotation->mutable_color()->set_r(0);
  annotation->mutable_color()->set_g(0);
  annotation->mutable_color()->set_b(0);
  
  rectangle->set_left(margin_left_right);
  rectangle->set_top(image_height - (box_height + box_margin_bottom));
  rectangle->set_bottom(image_height - box_margin_bottom);
  rectangle->set_right(image_width - margin_left_right);
 
  cc->Outputs()
    .Get(kRenderDataTag, 0)
    .AddPacket(mediapipe::MakePacket<RenderData>(background_render_data).At(cc->InputTimestamp()));
  cc->Outputs()
      .Get(kRenderDataTag, 1)
      .AddPacket(mediapipe::MakePacket<RenderData>(render_data).At(cc->InputTimestamp()));

  return ::mediapipe::OkStatus();
}

REGISTER_CALCULATOR(TextToRenderDataCalculator);

}  // namespace signlang

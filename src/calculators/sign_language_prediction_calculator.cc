
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/port/status.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

using namespace mediapipe;

namespace signlang
{

constexpr char kLandmarksTag[] = "NORM_LANDMARKS";
constexpr char kFaceDetectionsTag[] = "DETECTIONS";
constexpr char kTextOutputTag[] = "TEXT";
constexpr int maxFrames = 100;


// Example config:
// node {
//   calculator: "SignLanguagePredictionCalculator"
//   input_stream: "DETECTIONS:output_detections"
//   input_stream: "NORM_LANDMARKS:multi_hand_landmarks"
//   output_stream: "TEXT:prediction"
// }
class SignLanguagePredictionCalculator : public CalculatorBase
{
    public:
        static ::mediapipe::Status GetContract(CalculatorContract *cc);
        ::mediapipe::Status Open(CalculatorContext *cc) override;
        ::mediapipe::Status Process(CalculatorContext *cc) override;

    private:
        void AddFaceDetectionsTo(std::vector<float> &coordinates, CalculatorContext *cc);
        void AddMultiHandDetectionsTo(std::vector<float> &coordinates, CalculatorContext *cc);
        std::vector<std::vector<float>> frames;
        std::unique_ptr<tflite::FlatBufferModel> model;
        std::unique_ptr<tflite::Interpreter> interpreter;
};

::mediapipe::Status SignLanguagePredictionCalculator::GetContract(CalculatorContract *cc)
{
    RET_CHECK(cc->Inputs().HasTag(kLandmarksTag)) << "No input has the label " << kLandmarksTag << ".";
    RET_CHECK(cc->Inputs().HasTag(kFaceDetectionsTag)) << "No input has the label " << kFaceDetectionsTag << ".";
    cc->Inputs().Tag(kLandmarksTag).Set<std::vector<NormalizedLandmarkList>>();
    cc->Inputs().Tag(kFaceDetectionsTag).Set<std::vector<Detection>>();
    cc->Outputs().Tag(kTextOutputTag).Set<std::string>();
    return ::mediapipe::OkStatus();
}
::mediapipe::Status SignLanguagePredictionCalculator::Open(CalculatorContext *cc) {
    frames = {};
    const char* filename = "models/sign_lang_recognition.tflite";
    // Load the model
    model = tflite::FlatBufferModel::BuildFromFile(filename);
    RET_CHECK(model != nullptr) << "Building model from " << filename << " failed.";
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    //interpreter->ResizeInputTensor(0, {100, 86});
    interpreter->AllocateTensors();
    tflite::PrintInterpreterState(interpreter.get());
    LOG(INFO) << "tensors size: " << interpreter->tensors_size() << "\n";
    LOG(INFO) << "nodes size: " << interpreter->nodes_size() << "\n";
    LOG(INFO) << "inputs: " << interpreter->inputs().size() << "\n";
    LOG(INFO) << "input(0) name: " << interpreter->GetInputName(0) << "\n";
    return ::mediapipe::OkStatus();
}

::mediapipe::Status SignLanguagePredictionCalculator::Process(CalculatorContext *cc)
{
    LOG(INFO) << "Processing started!";
    std::vector<float> coordinates = {};
    AddFaceDetectionsTo(coordinates, cc);
    if (coordinates.size() == 0) { // No face detected.        
        coordinates.push_back(0.0); // 0 face_x
        coordinates.push_back(0.0); // 0 face_y
    }
    AddMultiHandDetectionsTo(coordinates, cc);
    if (coordinates.size() != 44 && coordinates.size() != 86) {
        LOG(WARNING) << "Expected coordinates to have a size of 44 or 86. Actual size: " << coordinates.size();
    }
    if (frames.size() >= maxFrames) {
        frames.erase(frames.begin());
    }

    while (frames.size() < (maxFrames - 1)) {
        std::vector<float> frame = {};
        for (size_t i = 0; i < 86; i++)
        {
            frame.push_back(0.0F);
        }
        frames.push_back(frame);
    }
    
    frames.push_back(coordinates);
    LOG(INFO) << "Frames size: " << frames.size();
    LOG(INFO) << "First frame size: " << frames[0].size();
    int input = interpreter->inputs()[0];
    TfLiteIntArray* dims = interpreter->tensor(input)->dims;
    LOG(INFO) << "Shape: {" << dims->data[0] << ", " << dims->data[1] << "}";
    float* input_data_ptr = interpreter->typed_tensor<float>(0);
    RET_CHECK(input_data_ptr != nullptr);
    size_t x = 0;
    for (size_t i = 0; i < frames.size(); i++)
    {
        // LOG(INFO) << "Frame: " << i;
        std::vector<float> frame = frames[i];
        for (size_t j = 0; j < frame.size(); j++)
        {
            // LOG(INFO) << "Coordinate: " << j;
            *(input_data_ptr) = frames[i][j];
            input_data_ptr++;
            x++;
        }
        x++;
    }
    interpreter->Invoke();
    int output_idx = interpreter->outputs()[0];
    float* output = interpreter->typed_output_tensor<float>(0);
    float predictions[12] = {};
    int highest_pred_idx = -1;
    float highest_pred = 0.0F;
    for (size_t i = 0; i < 12; i++)
    {
        LOG(INFO) << "OUTPUT (" << i << "): " << *output;
        predictions[i] = *output;
        if (*output > highest_pred) {
            highest_pred = *output;
            highest_pred_idx = i;
        }
        *output++;
    }
    // Here takes the prediction place!
    std::string prediction = std::to_string(highest_pred);
    LOG(INFO) << prediction;
    cc->Outputs()
    .Tag(kTextOutputTag)
    .AddPacket(mediapipe::MakePacket<std::string>(prediction)
    .At(cc->InputTimestamp()));
    return ::mediapipe::OkStatus();
}

REGISTER_CALCULATOR(SignLanguagePredictionCalculator);

void SignLanguagePredictionCalculator::AddFaceDetectionsTo(std::vector<float> &coordinates, CalculatorContext *cc)
{
    const std::vector<Detection> &faceDetections =
        cc->Inputs().Tag(kFaceDetectionsTag).Get<std::vector<Detection>>();
    
    if (!faceDetections.size()) { return; }

    const Detection &face = faceDetections[0];
    LocationData locationData = face.location_data();
    int kpSize = locationData.relative_keypoints_size();
    
    if (!kpSize) { return; }
    
    float faceX = face.location_data().relative_keypoints(0).x();
    coordinates.push_back(faceX);
    coordinates.push_back(face.location_data().relative_keypoints(0).y());
}

void SignLanguagePredictionCalculator::AddMultiHandDetectionsTo(std::vector<float> &coordinates, CalculatorContext *cc)
{
    const std::vector<NormalizedLandmarkList> &multiHandLandmarks =
        cc->Inputs().Tag(kLandmarksTag).Get<std::vector<NormalizedLandmarkList>>();
    
    for (NormalizedLandmarkList landmarks : multiHandLandmarks)
    {
        for (int i = 0; i < landmarks.landmark_size(); ++i)
        {
            const NormalizedLandmark &landmark = landmarks.landmark(i);
            coordinates.push_back(landmark.x());
            coordinates.push_back(landmark.y());
        }
    }
}

} // namespace signlang

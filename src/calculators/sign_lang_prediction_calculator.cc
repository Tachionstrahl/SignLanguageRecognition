
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
constexpr char kLabelsSidePacketTag[] = "LABELS";
constexpr int maxFrames = 100;
constexpr int thresholdFramesCount = 10;
constexpr int minFramesForInference = 10;
constexpr float defaultPoint = 0.0F;
constexpr char tfLiteModelPath[] = "models/sign_lang_recognition.tflite";
constexpr bool verbose = true;


// Example config:
// node {
//   calculator: "SignLangPredictionCalculator"
//   input_stream: "DETECTIONS:output_detections"
//   input_stream: "NORM_LANDMARKS:multi_hand_landmarks"
//   output_stream: "TEXT:prediction"
// }
class SignLangPredictionCalculator : public CalculatorBase
{
    public:
        static ::mediapipe::Status GetContract(CalculatorContract *cc);
        ::mediapipe::Status Open(CalculatorContext *cc) override;
        ::mediapipe::Status Process(CalculatorContext *cc) override;

    private:
        void AddFaceDetectionsTo(std::vector<float> &coordinates, CalculatorContext *cc);
        void AddMultiHandDetectionsTo(std::vector<float> &coordinates, CalculatorContext *cc);
        ::mediapipe::Status UpdateFrames(CalculatorContext *cc);
        bool ShouldPredict(); 
        ::mediapipe::Status FillInputTensor();
        void SetOutput(const std::string *str, ::mediapipe::CalculatorContext *cc);
        void DeleteFramesBuffer();
        std::vector<std::vector<float>> frames = {};
        std::unique_ptr<tflite::FlatBufferModel> model;
        std::unique_ptr<tflite::Interpreter> interpreter;
        std::string outputText = "Waiting...";
        std::vector<std::string> labelMap = {};
        int framesSinceLastPrediction = 0;
        int emptyFrames = 0;
};

::mediapipe::Status SignLangPredictionCalculator::GetContract(CalculatorContract *cc)
{
    RET_CHECK(cc->InputSidePackets().HasTag(kLabelsSidePacketTag)) << "Missing " << kLabelsSidePacketTag << " input side packet";
    RET_CHECK(cc->Inputs().HasTag(kLandmarksTag)) << "No input has the label " << kLandmarksTag;
    RET_CHECK(cc->Inputs().HasTag(kFaceDetectionsTag)) << "No input has the label " << kFaceDetectionsTag;
    cc->InputSidePackets().Tag(kLabelsSidePacketTag).Set<std::string>();
    cc->Inputs().Tag(kLandmarksTag).Set<std::vector<NormalizedLandmarkList>>();
    cc->Inputs().Tag(kFaceDetectionsTag).Set<std::vector<Detection>>();
    cc->Outputs().Tag(kTextOutputTag).Set<std::string>();
    return ::mediapipe::OkStatus();
}
::mediapipe::Status SignLangPredictionCalculator::Open(CalculatorContext *cc) {
    // Get Labels
    std::stringstream labels(cc->InputSidePackets().Tag(kLabelsSidePacketTag).Get<std::string>());
    std::string nextLabel;
    while(std::getline(labels, nextLabel, '\n')) {
        labelMap.push_back(nextLabel);
    }
    // Load the model
    model = tflite::FlatBufferModel::BuildFromFile(tfLiteModelPath);
    RET_CHECK(model != nullptr) << "Building model from " << tfLiteModelPath << " failed.";
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

::mediapipe::Status SignLangPredictionCalculator::Process(CalculatorContext *cc)
{
    // LOG(INFO) << "Processing started!";
    
    RET_CHECK_OK(UpdateFrames(cc));
    if (!ShouldPredict()) {
        std::string text = "Buffered: " + std::to_string(framesSinceLastPrediction);
        SetOutput(&text, cc);
        return ::mediapipe::OkStatus();
    }

    // Fill frames up to maximum
    while (frames.size() < (maxFrames)) {
        std::vector<float> frame = {};
        for (size_t i = 0; i < 86; i++)
        {
            frame.push_back(defaultPoint);
        }
        frames.push_back(frame);
    }
    LOG(INFO) << "Frames: " << frames.size();
    RET_CHECK_OK(FillInputTensor());

    interpreter->Invoke();

    int output_idx = interpreter->outputs()[0];
    float* output = interpreter->typed_tensor<float>(output_idx);
    int highest_pred_idx = -1;
    float highest_pred = 0.0F;
    for (size_t i = 0; i < 12; i++)
    {
        LOG(INFO) << "OUTPUT (" << i << "): " << *output;
        if (*output > highest_pred) {
            highest_pred = *output;
            highest_pred_idx = i;
        }
        *output++;
    }
    RET_CHECK_GT(highest_pred_idx, -1) << "No prediction found.";

    std::string prediction = labelMap[highest_pred_idx] + ", " + std::to_string(highest_pred);
    outputText = prediction;
    LOG(INFO) << "Predicted: " << outputText;
    SetOutput(&outputText, cc);
    DeleteFramesBuffer();
    return ::mediapipe::OkStatus();
}

void SignLangPredictionCalculator::DeleteFramesBuffer() {
    framesSinceLastPrediction = 0;
    emptyFrames = 0;
    frames.clear();
}

void SignLangPredictionCalculator::SetOutput(const std::string *str, ::mediapipe::CalculatorContext *cc) {
    cc->Outputs()
    .Tag(kTextOutputTag)
    .AddPacket(mediapipe::MakePacket<std::string>(*str)
    .At(cc->InputTimestamp()));
}

::mediapipe::Status SignLangPredictionCalculator::FillInputTensor() {
    int input = interpreter->inputs()[0];
    TfLiteIntArray* dims = interpreter->tensor(input)->dims;
    LOG(INFO) << "Shape: {" << dims->data[0] << ", " << dims->data[1] << "}";
    float* input_data_ptr = interpreter->typed_input_tensor<float>(0);
    RET_CHECK(input_data_ptr != nullptr);
    for (size_t i = 0; i < frames.size(); i++)
    {
        // LOG(INFO) << "Frame: " << i;
        std::vector<float> frame = frames[i];
        for (size_t j = 0; j < frame.size(); j++)
        {
            // LOG(INFO) << "Coordinate: " << j;
            *(input_data_ptr) = frames[i][j];
            input_data_ptr++;
        }
    }
    return ::mediapipe::OkStatus();
}

::mediapipe::Status SignLangPredictionCalculator::UpdateFrames(CalculatorContext *cc) {
    std::vector<float> coordinates = {};
    
    AddFaceDetectionsTo(coordinates, cc);
    if (coordinates.size() == 0) { // No face detected.        
        coordinates.push_back(defaultPoint); // 0 face_x
        coordinates.push_back(defaultPoint); // 0 face_y
    }
    AddMultiHandDetectionsTo(coordinates, cc);

    if (coordinates.size() < 44) { // No hands detected
        emptyFrames++;
        return ::mediapipe::OkStatus();
    }
    emptyFrames = 0;
    while (coordinates.size() < 86) {
        coordinates.push_back(defaultPoint);
    }
    
    RET_CHECK_EQ(coordinates.size(), 86) << "Coordinates size not equal 86. Actual size: " << coordinates.size();
    
    if (frames.size() >= maxFrames) {
        frames.erase(frames.begin());
    }

    // Put actual frame into array.
    frames.push_back(coordinates);
    framesSinceLastPrediction++;
    return ::mediapipe::OkStatus();
}

bool SignLangPredictionCalculator::ShouldPredict() {
    // Minimum frames required for inference
    if (framesSinceLastPrediction < minFramesForInference) {
        return false;
    }
    // Long enough without hands to predict.
    if (emptyFrames >= thresholdFramesCount) {
        return true;
    }
    return false;
}

void SignLangPredictionCalculator::AddFaceDetectionsTo(std::vector<float> &coordinates, CalculatorContext *cc)
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

void SignLangPredictionCalculator::AddMultiHandDetectionsTo(std::vector<float> &coordinates, CalculatorContext *cc)
{
    const std::vector<NormalizedLandmarkList> multiHandLandmarks =
        cc->Inputs().Tag(kLandmarksTag).Get<std::vector<NormalizedLandmarkList>>();
    for (NormalizedLandmarkList landmarks : multiHandLandmarks)
    {
        for (int i = 0; i < landmarks.landmark_size(); ++i)
        {
            const NormalizedLandmark &landmark = landmarks.landmark(i);
            if (landmark.x() == 0 && landmark.y() == 0){
                continue;
            }
            coordinates.push_back(landmark.x());
            coordinates.push_back(landmark.y());
        }
    }
}

REGISTER_CALCULATOR(SignLangPredictionCalculator);

} // namespace signlang

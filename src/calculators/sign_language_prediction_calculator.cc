
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/port/status.h"

using namespace mediapipe;

namespace signlang
{

constexpr char kLandmarksTag[] = "NORM_LANDMARKS";
constexpr char kFaceDetectionsTag[] = "DETECTIONS";
constexpr char kTextOutputTag[] = "TEXT";

// Takes in a std::string, draws the text std::string by cv::putText(), and
// outputs an ImageFrame.
//
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
        ::mediapipe::Status Process(CalculatorContext *cc) override;

    private:
        void AddFaceDetectionsTo(std::vector<float> &coordinates, CalculatorContext *cc);
        void AddMultiHandDetectionsTo(std::vector<float> &coordinates, CalculatorContext *cc);
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

::mediapipe::Status SignLanguagePredictionCalculator::Process(CalculatorContext *cc)
{

    std::vector<float> coordinates = {};
    AddFaceDetectionsTo(coordinates, cc);
    if (coordinates.size() == 0) { // No face detected.        
        coordinates.push_back(0.0); // 0 face_x
        coordinates.push_back(0.0); // 0 face_y
    }

    AddMultiHandDetectionsTo(coordinates, cc);

    if (coordinates.size() != 44 && coordinates.size() != 86) {
        //LOG(WARNING) << "Expected coordinates to have a size of 44 or 86. Actual size: " << coordinates.size();
        // return mediapipe::OkStatus();
    }

    // Here takes the prediction place!
    std::string prediction = "Wort!";
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

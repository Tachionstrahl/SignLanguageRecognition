// Calculator für das Schreiben von Kopf- und Hand-Tracking-Punkten in eine CSV Datei.
//POMMES
#include <cstdio>
#include <iostream>
#include <fstream>
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/detection.pb.h"
using namespace std;
using namespace mediapipe;
namespace signlang
{
constexpr char kLandmarksTag[] = "NORM_LANDMARKS";
constexpr char kFaceDetectionsTag[] = "DETECTIONS";
constexpr char kCSVPathTag[] = "CSV_OUTPUT_FILE_PATH";

constexpr bool relative = false;
constexpr bool use3D = false;
constexpr bool reduceLandmarks = false;
constexpr char csvHeader2D[] = "face_x;face_y;landmark_x_1;landmark_y_1;landmark_x_2;landmark_y_2;landmark_x_3;landmark_y_3;landmark_x_4;landmark_y_4;landmark_x_5;landmark_y_5;landmark_x_6;landmark_y_6;landmark_x_7;landmark_y_7;landmark_x_8;landmark_y_8;landmark_x_9;landmark_y_9;landmark_x_10;landmark_y_10;landmark_x_11;landmark_y_11;landmark_x_12;landmark_y_12;landmark_x_13;landmark_y_13;landmark_x_14;landmark_y_14;landmark_x_15;landmark_y_15;landmark_x_16;landmark_y_16;landmark_x_17;landmark_y_17;landmark_x_18;landmark_y_18;landmark_x_19;landmark_y_19;landmark_x_20;landmark_y_20;landmark_x_21;landmark_y_21;landmark_x_22;landmark_y_22;landmark_x_23;landmark_y_23;landmark_x_24;landmark_y_24;landmark_x_25;landmark_y_25;landmark_x_26;landmark_y_26;landmark_x_27;landmark_y_27;landmark_x_28;landmark_y_28;landmark_x_29;landmark_y_29;landmark_x_30;landmark_y_30;landmark_x_31;landmark_y_31;landmark_x_32;landmark_y_32;landmark_x_33;landmark_y_33;landmark_x_34;landmark_y_34;landmark_x_35;landmark_y_35;landmark_x_36;landmark_y_36;landmark_x_37;landmark_y_37;landmark_x_38;landmark_y_38;landmark_x_39;landmark_y_39;landmark_x_40;landmark_y_40;landmark_x_41;landmark_y_41;landmark_x_42;landmark_y_42";
constexpr char csvHeader3D[] = "face_x;face_y;landmark_x_1;landmark_y_1;landmark_z_1;landmark_x_2;landmark_y_2;landmark_z_2;landmark_x_3;landmark_y_3;landmark_z_3;landmark_x_4;landmark_y_4;landmark_z_4;landmark_x_5;landmark_y_5;landmark_z_5;landmark_x_6;landmark_y_6;landmark_z_6;landmark_x_7;landmark_y_7;landmark_z_7;landmark_x_8;landmark_y_8;landmark_z_8;landmark_x_9;landmark_y_9;landmark_z_9;landmark_x_10;landmark_y_10;landmark_z_10;landmark_x_11;landmark_y_11;landmark_z_11;landmark_x_12;landmark_y_12;landmark_z_12;landmark_x_13;landmark_y_13;landmark_z_13;landmark_x_14;landmark_y_14;landmark_z_14;landmark_x_15;landmark_y_15;landmark_z_15;landmark_x_16;landmark_y_16;landmark_z_16;landmark_x_17;landmark_y_17;landmark_z_17;landmark_x_18;landmark_y_18;landmark_z_18;landmark_x_19;landmark_y_19;landmark_z_19;landmark_x_20;landmark_y_20;landmark_z_20;landmark_x_21;landmark_y_21;landmark_z_21;landmark_x_22;landmark_y_22;landmark_z_22;landmark_x_23;landmark_y_23;landmark_z_23;landmark_x_24;landmark_y_24;landmark_z_24;landmark_x_25;landmark_y_25;landmark_z_25;landmark_x_26;landmark_y_26;landmark_z_26;landmark_x_27;landmark_y_27;landmark_z_27;landmark_x_28;landmark_y_28;landmark_z_28;landmark_x_29;landmark_y_29;landmark_z_29;landmark_x_30;landmark_y_30;landmark_z_30;landmark_x_31;landmark_y_31;landmark_z_31;landmark_x_32;landmark_y_32;landmark_z_32;landmark_x_33;landmark_y_33;landmark_z_33;landmark_x_34;landmark_y_34;landmark_z_34;landmark_x_35;landmark_y_35;landmark_z_35;landmark_x_36;landmark_y_36;landmark_z_36;landmark_x_37;landmark_y_37;landmark_z_37;landmark_x_38;landmark_y_38;landmark_z_38;landmark_x_39;landmark_y_39;landmark_z_39;landmark_x_40;landmark_y_40;landmark_z_40;landmark_x_41;landmark_y_41;landmark_z_41;landmark_x_42;landmark_y_42;landmark_z_42";
constexpr char csvHeader2DReduced[] = "face_x;face_y;landmark_x_1;landmark_y_1;landmark_x_2;landmark_y_2;landmark_x_3;landmark_y_3;landmark_x_4;landmark_y_4;landmark_x_5;landmark_y_5;landmark_x_6;landmark_y_6;landmark_x_7;landmark_y_7;landmark_x_8;landmark_y_8;landmark_x_9;landmark_y_9;landmark_x_10;landmark_y_10;landmark_x_11;landmark_y_11;landmark_x_12;landmark_y_12";
constexpr char csvHeader3DReduced[] = "face_x;face_y;landmark_x_1;landmark_y_1;landmark_z_1;landmark_x_2;landmark_y_2;landmark_z_2;landmark_x_3;landmark_y_3;landmark_z_3;landmark_x_4;landmark_y_4;landmark_z_4;landmark_x_5;landmark_y_5;landmark_z_5;landmark_x_6;landmark_y_6;landmark_z_6;landmark_x_7;landmark_y_7;landmark_z_7;landmark_x_8;landmark_y_8;landmark_z_8;landmark_x_9;landmark_y_9;landmark_z_9;landmark_x_10;landmark_y_10;landmark_z_10;landmark_x_11;landmark_y_11;landmark_z_11;landmark_x_12;landmark_y_12;landmark_z_12";
class DetectionsToCSVCalculator : public CalculatorBase
{
    public:
        static Status GetContract(CalculatorContract *cc);
        Status Process(CalculatorContext *cc) final;
        Status Open(CalculatorContext *cc);
        Status Close(CalculatorContext *cc) final;
    private:
        std::vector<float> GetCoordinatesRelative(std::vector<float> coordinatesB);
        std::fstream csvFile;
        std::vector<float> coordinatesA = {};
        int relevantLandmarks [12] = {0,1,2,4,5,8,9,12,13,16,17,20};
};

Status DetectionsToCSVCalculator::GetContract(CalculatorContract *cc)
{
    RET_CHECK(cc->Inputs().HasTag(kLandmarksTag)) << "No input has the label " << kLandmarksTag << ".";
    RET_CHECK(cc->Inputs().HasTag(kFaceDetectionsTag)) << "No input has the label " << kFaceDetectionsTag << ".";
    cc->Inputs().Tag(kLandmarksTag).Set<std::vector<NormalizedLandmarkList>>();
    cc->Inputs().Tag(kFaceDetectionsTag).Set<std::vector<Detection>>();
    cc->InputSidePackets().Tag(kCSVPathTag).Set<std::string>();
    return OkStatus();
}

Status DetectionsToCSVCalculator::Open(CalculatorContext *cc)
{
    if (csvFile.is_open()) {
        csvFile.flush();
        csvFile.close();
    }
    const std::string &file_path =
        cc->InputSidePackets().Tag("CSV_OUTPUT_FILE_PATH").Get<std::string>();
    csvFile.open(file_path, fstream::out);
    if (use3D) {
         if (reduceLandmarks) {
            csvFile << csvHeader3DReduced << endl;
        } else {
            csvFile << csvHeader3D << endl;
        }   
    } else {
        if (reduceLandmarks) {
            csvFile << csvHeader2DReduced << endl;
        } else {
            csvFile << csvHeader2D << endl;
        }        
    }
    return OkStatus();
}
Status DetectionsToCSVCalculator::Process(CalculatorContext *cc)
{
    std::vector<float> coordinates = {};
    const std::vector<Detection> &faceDetections =
        cc->Inputs().Tag(kFaceDetectionsTag).Get<std::vector<Detection>>();
    if (!faceDetections.size())
    {
        return OkStatus();
    }
    const Detection &face = faceDetections[0];
    LocationData locationData = face.location_data();
    int kpSize = locationData.relative_keypoints_size();
    if (!kpSize)
    {
        return OkStatus();
    }
    float faceX = face.location_data().relative_keypoints(0).x();
    coordinates.push_back(faceX);
    coordinates.push_back(face.location_data().relative_keypoints(0).y());
    const std::vector<NormalizedLandmarkList> &multiHandLandmarks =
        cc->Inputs().Tag(kLandmarksTag).Get<std::vector<NormalizedLandmarkList>>();
    for (NormalizedLandmarkList landmarks : multiHandLandmarks)
    {
        for (int i = 0; i < landmarks.landmark_size(); ++i)
        {
            if (reduceLandmarks) {
                int *lIdx = std::find(std::begin(relevantLandmarks), std::end(relevantLandmarks), i);
                // If the element is not found, std::find returns the end of the range
                if (lIdx == std::end(relevantLandmarks)) {
                    continue;
                }
            }
            const NormalizedLandmark &landmark = landmarks.landmark(i);
            auto x = landmark.x();
            auto y = landmark.y();
            auto z = landmark.z();

            if (x != 0 && y != 0) {
                coordinates.push_back(landmark.x());
                coordinates.push_back(landmark.y());
            }

            if (use3D && z != 0) {
                coordinates.push_back(landmark.z());
            }
        }
    }

    int landmarksPerHand = reduceLandmarks ? 12 : 21;
    int dimensions = use3D ? 3 : 2;
    int faceLandmarkPoints = 2;
    if (coordinates.size() == landmarksPerHand * dimensions + faceLandmarkPoints || 
    coordinates.size() == landmarksPerHand * dimensions * 2 + faceLandmarkPoints)
    {
        if (relative) {
            coordinates = GetCoordinatesRelative(coordinates);
        }
        if (coordinates.size() <= 0) {
            return OkStatus();
        }
        for (int i = 0; i < coordinates.size(); i++)
        {
            csvFile << coordinates[i];
            if (i < coordinates.size() - 1)
            {
                csvFile << ";";
            }
        }
        csvFile << endl;
    } else {
        LOG(ERROR) << "Landmarks size not expected: " << coordinates.size();
    }
    coordinates.clear();
    return OkStatus();
}

Status DetectionsToCSVCalculator::Close(CalculatorContext *cc)
{
    csvFile.flush();
    csvFile.close();
    return OkStatus();
}

std::vector<float> DetectionsToCSVCalculator::GetCoordinatesRelative(std::vector<float> coordinatesB) {
    if (coordinatesA.size() <= 0) {
        coordinatesA = coordinatesB;
        return {};
    }
    std::vector<float> relativeCoordinates = {};
    for (size_t i = 0; i < coordinatesB.size(); i++)
    {
        if (coordinatesA.size() >= i+1) {
            float delta = coordinatesB[i] - coordinatesA[i];
            int change;
            if (delta > 0.001) {
                change = 1;
            } else if (delta < -0.001)
            {
                change = -1;
            } else {
                change = 0;
            }
            
            relativeCoordinates.push_back(change);
        }
    }
    coordinatesA = coordinatesB;
    return relativeCoordinates;
}


REGISTER_CALCULATOR(DetectionsToCSVCalculator);
} // namespace signlang

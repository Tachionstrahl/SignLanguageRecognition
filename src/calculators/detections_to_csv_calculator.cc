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
constexpr char csvHeader[] = "face_x;face_y;landmark_x_1;landmark_y_1;landmark_x_2;landmark_y_2;landmark_x_3;landmark_y_3;landmark_x_4;landmark_y_4;landmark_x_5;landmark_y_5;landmark_x_6;landmark_y_6;landmark_x_7;landmark_y_7;landmark_x_8;landmark_y_8;landmark_x_9;landmark_y_9;landmark_x_10;landmark_y_10;landmark_x_11;landmark_y_11;landmark_x_12;landmark_y_12;landmark_x_13;landmark_y_13;landmark_x_14;landmark_y_14;landmark_x_15;landmark_y_15;landmark_x_16;landmark_y_16;landmark_x_17;landmark_y_17;landmark_x_18;landmark_y_18;landmark_x_19;landmark_y_19;landmark_x_20;landmark_y_20;landmark_x_21;landmark_y_21;landmark_x_22;landmark_y_22;landmark_x_23;landmark_y_23;landmark_x_24;landmark_y_24;landmark_x_25;landmark_y_25;landmark_x_26;landmark_y_26;landmark_x_27;landmark_y_27;landmark_x_28;landmark_y_28;landmark_x_29;landmark_y_29;landmark_x_30;landmark_y_30;landmark_x_31;landmark_y_31;landmark_x_32;landmark_y_32;landmark_x_33;landmark_y_33;landmark_x_34;landmark_y_34;landmark_x_35;landmark_y_35;landmark_x_36;landmark_y_36;landmark_x_37;landmark_y_37;landmark_x_38;landmark_y_38;landmark_x_39;landmark_y_39;landmark_x_40;landmark_y_40;landmark_x_41;landmark_y_41;landmark_x_42;landmark_y_42";
class DetectionsToCSVCalculator : public CalculatorBase
{
public:
    std::fstream csvFile;
    static Status GetContract(CalculatorContract *cc)
    {
        puts("GetContract called.");
        RET_CHECK(cc->Inputs().HasTag(kLandmarksTag)) << "No input has the label " << kLandmarksTag << ".";
        RET_CHECK(cc->Inputs().HasTag(kFaceDetectionsTag)) << "No input has the label " << kFaceDetectionsTag << ".";
        cc->Inputs().Tag(kLandmarksTag).Set<std::vector<NormalizedLandmarkList>>();
        cc->Inputs().Tag(kFaceDetectionsTag).Set<std::vector<Detection>>();
        cc->InputSidePackets().Tag(kCSVPathTag).Set<std::string>();
        return OkStatus();
    }

    Status Open(CalculatorContext *cc) final
    {
        puts("Open called.");
        if (csvFile.is_open()) {
            csvFile.flush();
            csvFile.close();
        }
        const std::string &file_path =
            cc->InputSidePackets().Tag("CSV_OUTPUT_FILE_PATH").Get<std::string>();
        csvFile.open(file_path, fstream::out);
        csvFile << csvHeader << endl;
        return OkStatus();
    }

    Status Process(CalculatorContext *cc) final
    {
        puts("Process called");

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
        cout << "Anzahl Hände: " << multiHandLandmarks.size() << endl;
        for (NormalizedLandmarkList landmarks : multiHandLandmarks)
        {
            cout << "Landmark size: " << landmarks.landmark_size() << endl;
            for (int i = 0; i < landmarks.landmark_size(); ++i)
            {
                const NormalizedLandmark &landmark = landmarks.landmark(i);
                coordinates.push_back(landmark.x());
                coordinates.push_back(landmark.y());
                // cout << "X: " << landmark.x();
                // cout << "Y: " << landmark.y();
            }
        }

        if (coordinates.size() == 44 || coordinates.size() == 86)
        {
            for (int i = 0; i < coordinates.size(); i++)
            {
                cout << "Schreibe... [" << i << "]";
                csvFile << coordinates[i];
                if (i < coordinates.size() - 1)
                {
                    csvFile << ";";
                }
            }
            csvFile << endl;
        }
        else
        {
            cout << "Koordinaten: " << coordinates.size();
        }
        coordinates.clear();
        return OkStatus();
    }
    Status Close(CalculatorContext *cc) final
    {
        csvFile.flush();
        csvFile.close();
        return OkStatus();
    }
};

REGISTER_CALCULATOR(DetectionsToCSVCalculator);
} // namespace signlang

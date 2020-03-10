// Calculator für das Schreiben von Kopf- und Hand-Tracking-Punkten in eine CSV Datei.

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
constexpr char csvHeader[] = "face_x;face_y;landmark_x_1;landmark_y_1;landmark_x_2;landmark_y_2;landmark_x_3;landmark_y_3;landmark_x_4;landmark_y_4;landmark_x_5;landmark_y_5;landmark_x_6;landmark_y_6;landmark_x_7;landmark_y_7;landmark_x_8;landmark_y_8;landmark_x_9;landmark_y_9;landmark_x_10;landmark_y_10;landmark_x_11;landmark_y_11;landmark_x_12;landmark_y_12;landmark_x_13;landmark_y_13;landmark_x_14;landmark_y_14;landmark_x_15;landmark_y_15;landmark_x_16;landmark_y_16;landmark_x_17;landmark_y_17;landmark_x_18;landmark_y_18;landmark_x_19;landmark_y_19;landmark_x_20;landmark_y_20;landmark_x_21;landmark_y_21";
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
        return OkStatus();
    }

    Status Open(CalculatorContext *cc) final
    {
        puts("Open called.");
        csvFile.open("example.csv", fstream::out);
        csvFile << csvHeader << endl;
        return OkStatus();
    }

    Status Process(CalculatorContext *cc) final
    {
        puts("Process called");

        std::vector<float> coordinates = {};
        const std::vector<Detection> &faceDetections =
            cc->Inputs().Tag(kFaceDetectionsTag).Get<std::vector<Detection>>();
        const Detection &face = faceDetections[0];
        coordinates.push_back(face.location_data().relative_keypoints(0).x());
        coordinates.push_back(face.location_data().relative_keypoints(0).y());
        const std::vector<NormalizedLandmarkList> &multiHandLandmarks =
            cc->Inputs().Tag(kLandmarksTag).Get<std::vector<NormalizedLandmarkList>>();
        RET_CHECK_LE(multiHandLandmarks.size(), 2) << "Too much hands";
        cout << "Anzahl Hände: " << multiHandLandmarks.size() << endl;
        for (NormalizedLandmarkList landmarks : multiHandLandmarks)
        {
            cout << "Landmark size: " << landmarks.landmark_size() << endl;
            for (int i = 0; i < landmarks.landmark_size(); ++i)
            {
                const NormalizedLandmark &landmark = landmarks.landmark(i);
                coordinates.push_back(landmark.x());
                coordinates.push_back(landmark.y());
                cout << "X: " << landmark.x();
                cout << "Y: " << landmark.y();
            }
        }

        if (coordinates.size() == 44)
        {
            for (int i = 0; i < coordinates.size(); i++)
            {
                csvFile << coordinates[i];
                if (i < coordinates.size() - 1)
                {
                    csvFile << ";";
                }
            }
            csvFile << endl;
        }
        coordinates.clear();
        return OkStatus();
    }
    Status Close(CalculatorContext *cc) final
    {
        csvFile.close();
    }
};

REGISTER_CALCULATOR(DetectionsToCSVCalculator);
} // namespace signlang

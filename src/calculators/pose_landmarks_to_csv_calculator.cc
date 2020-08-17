// Calculator für das Schreiben von Kopf- und Hand-Tracking-Punkten in eine CSV Datei.
//POMMES
#include <cstdio>
#include <iostream>
#include <fstream>
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
using namespace std;
using namespace mediapipe;
namespace signlang
{
    constexpr char kLandmarksTag[] = "NORM_LANDMARKS";
    constexpr char kCSVPathTag[] = "CSV_OUTPUT_FILE_PATH";

    constexpr bool relative = false;
    constexpr char csvHeader2D[] = "nose_x;nose_y;right_eye_inner_x;right_eye_inner_y;right_eye_x;right_eye_y;right_eye_outer_x;right_eye_outer_y;left_eye_inner_x;left_eye_inner_y;left_eye_x;left_eye_y;left_eye_outer_x;left_eye_outer_y;right_ear_x;right_ear_y;left_ear_x;left_ear_y;mouth_right_x;mouth_right_y;mouth_left_x;mouth_left_y;right_shoulder_x;right_shoulder_y;left_shoulder_x;left_shoulder_y;right_elbow_x;right_elbow_y;left_elbow_x;left_elbow_y;right_wrist_x;right_wrist_y;left_wrist_x;left_wrist_y;right_pinky_1_x;right_pinky_1_y;left_pinky_1_x;left_pinky_1_y;right_index_1_x;right_index_1_y;left_index_1_x;left_index_1_y;right_thumb_2_x;right_thumb_2_y;left_thumb_2_x;left_thumb_2_y;right_hip_x;right_hip_y;left_hip_x;left_hip_y";
    constexpr char csvHeader3D[] = "nose_x;nose_y;nose_z;right_eye_inner_x;right_eye_inner_y;right_eye_inner_z;right_eye_x;right_eye_y;right_eye_z;right_eye_outer_x;right_eye_outer_y;right_eye_outer_z;left_eye_inner_x;left_eye_inner_y;left_eye_inner_z;left_eye_x;left_eye_y;left_eye_z;left_eye_outer_x;left_eye_outer_y;left_eye_outer_z;right_ear_x;right_ear_y;right_ear_z;left_ear_x;left_ear_y;left_ear_z;mouth_right_x;mouth_right_y;mouth_right_z;mouth_left_x;mouth_left_y;mouth_left_z;right_shoulder_x;right_shoulder_y;right_shoulder_z;left_shoulder_x;left_shoulder_y;left_shoulder_z;right_elbow_x;right_elbow_y;right_elbow_z;left_elbow_x;left_elbow_y;left_elbow_z;right_wrist_x;right_wrist_y;right_wrist_z;left_wrist_x;left_wrist_y;left_wrist_z;right_pinky_1_x;right_pinky_1_y;right_pinky_1_z;left_pinky_1_x;left_pinky_1_y;left_pinky_1_z;right_index_1_x;right_index_1_y;right_index_1_z;left_index_1_x;left_index_1_y;left_index_1_z;right_thumb_2_x;right_thumb_2_y;right_thumb_2_z;left_thumb_2_x;left_thumb_2_y;left_thumb_2_z;right_hip_x;right_hip_y;right_hip_z;left_hip_x;left_hip_y;left_hip_z";
    class PoseLandmarksToCSVCalculator : public CalculatorBase
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
    };

    Status PoseLandmarksToCSVCalculator::GetContract(CalculatorContract *cc)
    {
        RET_CHECK(cc->Inputs().HasTag(kLandmarksTag)) << "No input has the label " << kLandmarksTag << ".";
        cc->Inputs().Tag(kLandmarksTag).Set<NormalizedLandmarkList>();
        cc->InputSidePackets().Tag(kCSVPathTag).Set<std::string>();
        return OkStatus();
    }

    Status PoseLandmarksToCSVCalculator::Open(CalculatorContext *cc)
    {
        if (csvFile.is_open())
        {
            csvFile.flush();
            csvFile.close();
        }
        const std::string &file_path =
            cc->InputSidePackets().Tag("CSV_OUTPUT_FILE_PATH").Get<std::string>();
        csvFile.open(file_path, fstream::out);
        csvFile << csvHeader2D << endl;
        return OkStatus();
    }
    Status PoseLandmarksToCSVCalculator::Process(CalculatorContext *cc)
    {
        try
        {
            std::vector<float> coordinates = {};
            const NormalizedLandmarkList &poseLandmarks =
                cc->Inputs().Tag(kLandmarksTag).Get<NormalizedLandmarkList>();
            for (int i = 0; i < 25; ++i)
            {
                const NormalizedLandmark &landmark = poseLandmarks.landmark(i);
                
                coordinates.push_back(landmark.x());
                coordinates.push_back(landmark.y());
                //coordinates.push_back(landmark.z());
            }
            if (coordinates.size() == 25*2) // 25 landmarks with x and y
            {
                if (relative)
                {
                    coordinates = GetCoordinatesRelative(coordinates);
                }
                if (coordinates.size() <= 0)
                {
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
            }
            else
            {
                LOG(INFO) << "Coordinates size is " << coordinates.size() << ". No frame written.";
            }
            coordinates.clear();
            return OkStatus();
        }
        catch (const std::exception &e)
        {
            LOG(ERROR) << e.what();
            return OkStatus();
        }
    }

    Status PoseLandmarksToCSVCalculator::Close(CalculatorContext *cc)
    {

        csvFile.flush();
        csvFile.close();
        return OkStatus();
    }

    std::vector<float> PoseLandmarksToCSVCalculator::GetCoordinatesRelative(std::vector<float> coordinatesB)
    {
        if (coordinatesA.size() <= 0)
        {
            coordinatesA = coordinatesB;
            return {};
        }
        std::vector<float> relativeCoordinates = {};
        for (size_t i = 0; i < coordinatesB.size(); i++)
        {
            if (coordinatesA.size() >= i + 1)
            {
                float delta = coordinatesB[i] - coordinatesA[i];
                int change;
                if (delta > 0.001)
                {
                    change = 1;
                }
                else if (delta < -0.001)
                {
                    change = -1;
                }
                else
                {
                    change = 0;
                }

                relativeCoordinates.push_back(change);
            }
        }
        coordinatesA = coordinatesB;
        return relativeCoordinates;
    }

    REGISTER_CALCULATOR(PoseLandmarksToCSVCalculator);
} // namespace signlang

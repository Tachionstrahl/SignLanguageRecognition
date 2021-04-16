
#include "calculators/sign_lang_prediction_calculator.pb.h"
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
#include <iostream>
#include <iomanip>
#include <ctime>
#include <cstdio>
#include <fstream>
#include <math.h>
#include <chrono>
#include <future>
#include <chrono>
#include <thread>

using namespace mediapipe;

namespace signlang
{
    constexpr char kPoseLandmarksTag[] = "POSE_LANDMARKS";
    constexpr char kLandmarksTag[] = "NORM_LANDMARKS";
    constexpr char kFaceDetectionsTag[] = "DETECTIONS";
    constexpr char kTextOutputTag[] = "TEXT";
    constexpr char kLabelsSidePacketTag[] = "LABELS";
    constexpr float defaultPoint = 0.0F;
    constexpr char csvHeader2D[] = "face_x;face_y;landmark_x_1;landmark_y_1;landmark_x_2;landmark_y_2;landmark_x_3;landmark_y_3;landmark_x_4;landmark_y_4;landmark_x_5;landmark_y_5;landmark_x_6;landmark_y_6;landmark_x_7;landmark_y_7;landmark_x_8;landmark_y_8;landmark_x_9;landmark_y_9;landmark_x_10;landmark_y_10;landmark_x_11;landmark_y_11;landmark_x_12;landmark_y_12;landmark_x_13;landmark_y_13;landmark_x_14;landmark_y_14;landmark_x_15;landmark_y_15;landmark_x_16;landmark_y_16;landmark_x_17;landmark_y_17;landmark_x_18;landmark_y_18;landmark_x_19;landmark_y_19;landmark_x_20;landmark_y_20;landmark_x_21;landmark_y_21;landmark_x_22;landmark_y_22;landmark_x_23;landmark_y_23;landmark_x_24;landmark_y_24;landmark_x_25;landmark_y_25;landmark_x_26;landmark_y_26;landmark_x_27;landmark_y_27;landmark_x_28;landmark_y_28;landmark_x_29;landmark_y_29;landmark_x_30;landmark_y_30;landmark_x_31;landmark_y_31;landmark_x_32;landmark_y_32;landmark_x_33;landmark_y_33;landmark_x_34;landmark_y_34;landmark_x_35;landmark_y_35;landmark_x_36;landmark_y_36;landmark_x_37;landmark_y_37;landmark_x_38;landmark_y_38;landmark_x_39;landmark_y_39;landmark_x_40;landmark_y_40;landmark_x_41;landmark_y_41;landmark_x_42;landmark_y_42";
    constexpr char csvHeader3D[] = "face_x;face_y;landmark_x_1;landmark_y_1;landmark_z_1;landmark_x_2;landmark_y_2;landmark_z_2;landmark_x_3;landmark_y_3;landmark_z_3;landmark_x_4;landmark_y_4;landmark_z_4;landmark_x_5;landmark_y_5;landmark_z_5;landmark_x_6;landmark_y_6;landmark_z_6;landmark_x_7;landmark_y_7;landmark_z_7;landmark_x_8;landmark_y_8;landmark_z_8;landmark_x_9;landmark_y_9;landmark_z_9;landmark_x_10;landmark_y_10;landmark_z_10;landmark_x_11;landmark_y_11;landmark_z_11;landmark_x_12;landmark_y_12;landmark_z_12;landmark_x_13;landmark_y_13;landmark_z_13;landmark_x_14;landmark_y_14;landmark_z_14;landmark_x_15;landmark_y_15;landmark_z_15;landmark_x_16;landmark_y_16;landmark_z_16;landmark_x_17;landmark_y_17;landmark_z_17;landmark_x_18;landmark_y_18;landmark_z_18;landmark_x_19;landmark_y_19;landmark_z_19;landmark_x_20;landmark_y_20;landmark_z_20;landmark_x_21;landmark_y_21;landmark_z_21;landmark_x_22;landmark_y_22;landmark_z_22;landmark_x_23;landmark_y_23;landmark_z_23;landmark_x_24;landmark_y_24;landmark_z_24;landmark_x_25;landmark_y_25;landmark_z_25;landmark_x_26;landmark_y_26;landmark_z_26;landmark_x_27;landmark_y_27;landmark_z_27;landmark_x_28;landmark_y_28;landmark_z_28;landmark_x_29;landmark_y_29;landmark_z_29;landmark_x_30;landmark_y_30;landmark_z_30;landmark_x_31;landmark_y_31;landmark_z_31;landmark_x_32;landmark_y_32;landmark_z_32;landmark_x_33;landmark_y_33;landmark_z_33;landmark_x_34;landmark_y_34;landmark_z_34;landmark_x_35;landmark_y_35;landmark_z_35;landmark_x_36;landmark_y_36;landmark_z_36;landmark_x_37;landmark_y_37;landmark_z_37;landmark_x_38;landmark_y_38;landmark_z_38;landmark_x_39;landmark_y_39;landmark_z_39;landmark_x_40;landmark_y_40;landmark_z_40;landmark_x_41;landmark_y_41;landmark_z_41;landmark_x_42;landmark_y_42;landmark_z_42";

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
        ::mediapipe::Status LoadOptions(CalculatorContext *cc);
        void AddFaceDetectionsTo(std::vector<float> &coordinates, CalculatorContext *cc);
        void AddMultiHandDetectionsTo(std::vector<float> &coordinates, CalculatorContext *cc);
        void AddPoseLandmarks(std::vector<float> &coordinates, CalculatorContext *cc);
        ::mediapipe::Status UpdateFrames(CalculatorContext *cc);
        bool ShouldPredict();
        ::mediapipe::Status FillInputTensor(std::vector<std::vector<float>> localFrames);
        void SetOutput(const std::string *str, ::mediapipe::CalculatorContext *cc);
        void DoAfterInference();
        void WriteFramesToFile(std::vector<std::vector<float>> frames, std::string prediction);
        bool DoInference();
        std::vector<std::vector<float>> framesWindow = {};
        std::unique_ptr<tflite::FlatBufferModel> model;
        std::unique_ptr<tflite::Interpreter> interpreter;
        std::tuple<std::string, float> outputWordProb = std::make_tuple("Waiting...", 1.0);
        std::vector<std::string> labelMap = {};
        std::vector<float> GetCoordinatesRelative(std::vector<float> coordinatesB);
        std::vector<float> coordinatesA = {};
        int framesSinceLastPrediction = 0;
        int emptyFrames = 0;
        // Options
        bool verboseLog = false;
        int framesWindowSize = 0;
        int thresholdFramesCount = 0;
        int minFramesForInference = 0;
        bool use3D = false;
        bool usePoseLandmarks = false;
        float probabilitityThreshold = 0.5;
        bool fluentPrediction = false;
        bool useRelative = false;
        std::string tfLiteModelPath;
        std::unique_ptr<std::future<bool>> inferenceFuture;
    };

    ::mediapipe::Status SignLangPredictionCalculator::GetContract(CalculatorContract *cc)
    {
        RET_CHECK(cc->InputSidePackets().HasTag(kLabelsSidePacketTag)) << "Missing " << kLabelsSidePacketTag << " input side packet";
        cc->InputSidePackets().Tag(kLabelsSidePacketTag).Set<std::string>();

        if (cc->Inputs().HasTag(kPoseLandmarksTag))
        {
            cc->Inputs().Tag(kPoseLandmarksTag).Set<NormalizedLandmarkList>();
        }
        else
        {
            RET_CHECK(cc->Inputs().HasTag(kLandmarksTag)) << "No input has the label " << kLandmarksTag;
            RET_CHECK(cc->Inputs().HasTag(kFaceDetectionsTag)) << "No input has the label " << kFaceDetectionsTag;
            cc->Inputs().Tag(kLandmarksTag).Set<std::vector<NormalizedLandmarkList>>();
            cc->Inputs().Tag(kFaceDetectionsTag).Set<std::vector<Detection>>();
        }
        cc->Outputs().Index(0).Set<std::tuple<std::string, float>>();
        return ::mediapipe::OkStatus();
    }
    ::mediapipe::Status SignLangPredictionCalculator::Open(CalculatorContext *cc)
    {
        // LOG(INFO) << "Open";
        if (cc->Inputs().HasTag(kPoseLandmarksTag))
        {
            usePoseLandmarks = true;
        }
        MP_RETURN_IF_ERROR(LoadOptions(cc)) << "Loading options failed";
        // Get Labels
        std::stringstream labels(cc->InputSidePackets().Tag(kLabelsSidePacketTag).Get<std::string>());
        std::string nextLabel;
        while (std::getline(labels, nextLabel, '\n'))
        {
            labelMap.push_back(nextLabel);
        }
        // Load the model
        model = tflite::FlatBufferModel::BuildFromFile(tfLiteModelPath.c_str());

        RET_CHECK(model != nullptr) << "Building model from " << tfLiteModelPath << " failed.";
        tflite::ops::builtin::BuiltinOpResolver resolver;
        tflite::InterpreterBuilder(*model, resolver)(&interpreter);
        interpreter->AllocateTensors();
        if (verboseLog)
        {
            tflite::PrintInterpreterState(interpreter.get());
            LOG(INFO) << "tensors size: " << interpreter->tensors_size() << "\n";
            LOG(INFO) << "nodes size: " << interpreter->nodes_size() << "\n";
            LOG(INFO) << "inputs: " << interpreter->inputs().size() << "\n";
            LOG(INFO) << "outputs: " << interpreter->outputs().size() << "\n";
            LOG(INFO) << "input(0) name: " << interpreter->GetInputName(0) << "\n";
        }
        return ::mediapipe::OkStatus();
    }

    ::mediapipe::Status SignLangPredictionCalculator::Process(CalculatorContext *cc)
    {
        RET_CHECK_OK(UpdateFrames(cc)) << "Updating frames failed.";
        if (inferenceFuture != nullptr && inferenceFuture->wait_for(std::chrono::milliseconds(10)) == std::future_status::ready)
        {
            inferenceFuture = nullptr;
            int output_idx = interpreter->outputs()[0];
            float *output = interpreter->typed_tensor<float>(output_idx);
            int highest_pred_idx = -1;
            float highest_pred = 0.0F;
            for (size_t i = 0; i < labelMap.size(); i++)
            {
                if (verboseLog)
                {
                    LOG(INFO) << labelMap[i] << ": " << *output;
                }
                if (*output > highest_pred)
                {
                    highest_pred = *output;
                    highest_pred_idx = i;
                }
                *output++;
            }
            if (highest_pred > probabilitityThreshold)
            {
                std::string prediction = labelMap[highest_pred_idx];
                outputWordProb = std::make_tuple(prediction, highest_pred);
                cc->Outputs().Index(0).AddPacket(mediapipe::MakePacket<std::tuple<std::string, float>>(outputWordProb)
                                                     .At(cc->InputTimestamp()));
            }
            else
            {
                cc->Outputs().Index(0).AddPacket(mediapipe::MakePacket<std::tuple<std::string, float>>(std::make_tuple("<unknown>", -1.0)).At(cc->InputTimestamp()));
            }
            return mediapipe::OkStatus();
        }
        if (!ShouldPredict())
        {
            if (fluentPrediction)
            {
                cc->Outputs().Index(0).AddPacket(mediapipe::MakePacket<std::tuple<std::string, float>>().At(cc->InputTimestamp()));
            }
            else
            {
                cc->Outputs()
                    .Index(0)
                    .AddPacket(mediapipe::MakePacket<std::tuple<std::string, float>>(std::make_tuple("Buffer", float(framesWindow.size())))
                                   .At(cc->InputTimestamp()));
            }

            return ::mediapipe::OkStatus();
        }
        // Fill frames up to maximum
        std::vector<std::vector<float>> localFrames = {};
        while (localFrames.size() < 100)
        {
            if (framesWindow.size() > localFrames.size())
            {
                localFrames.push_back(framesWindow[localFrames.size()]);
            }
            else
            {
                std::vector<float> frame = {};
                for (size_t i = 0; i < use3D ? 128 : 86; i++)
                {
                    frame.push_back(defaultPoint);
                }
                localFrames.push_back(frame);
            }
        }
        if (inferenceFuture == nullptr)
        {
            RET_CHECK_OK(FillInputTensor(localFrames));
            inferenceFuture = std::make_unique<std::future<bool>>(std::async(std::launch::async, [this]() { return DoInference(); }));
            DoAfterInference();
             cc->Outputs().Index(0).AddPacket(mediapipe::MakePacket<std::tuple<std::string, float>>(std::make_tuple("Inference", -1.0)).At(cc->InputTimestamp()));
        }
        // WriteFramesToFile(localFrames, std::get<0>(outputWordProb));
        
        return ::mediapipe::OkStatus();
    }

    bool SignLangPredictionCalculator::DoInference()
    {
        auto start = std::chrono::high_resolution_clock::now();
        interpreter->Invoke();
        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = finish - start;
        LOG(INFO) << "Inference time: " << elapsed.count();
        return true;
    }

    void SignLangPredictionCalculator::WriteFramesToFile(std::vector<std::vector<float>> frames, std::string prediction)
    {
        std::fstream csvFile;

        auto t = std::time(nullptr);
        auto tm = *std::localtime(&t);
        std::ostringstream oss;
        oss << "/home/michi/ML/SignLanguageRecognition/lab/data/absolute/live/Hallo/" << prediction << std::put_time(&tm, "%d-%m-%Y %H-%M-%S") << ".csv";
        // oss << "/home/datagroup/Development/SignLanguageRecognition/lab/data/live/ich/" << prediction << std::put_time(&tm, "%d-%m-%Y %H-%M-%S") << ".csv";
        const std::string filePath = oss.str();
        LOG(INFO) << "Writing to file " << filePath << " ...";
        csvFile.open(filePath, std::fstream::out);
        if (use3D)
        {
            csvFile << csvHeader3D << std::endl;
        }
        else
        {
            csvFile << csvHeader2D << std::endl;
        }
        for (size_t i = 0; i < frames.size(); i++)
        {
            auto frame = frames[i];
            for (size_t j = 0; j < frame.size(); j++)
            {
                csvFile << frame[j];
                if (j < frame.size() - 1)
                {
                    csvFile << ";";
                }
            }
            csvFile << std::endl;
        }
        csvFile.flush();
        csvFile.close();
    }

    ::mediapipe::Status SignLangPredictionCalculator::LoadOptions(
        CalculatorContext *cc)
    {
        const auto &options = cc->Options<SignLangPredictionCalculatorOptions>();
        verboseLog = options.verbose();
        framesWindowSize = options.frameswindowsize();
        thresholdFramesCount = options.thresholdframescount();
        minFramesForInference = options.minframesforinference();
        use3D = options.use3d();
        probabilitityThreshold = options.probabilitythreshold();
        tfLiteModelPath = options.tflitemodelpath();
        fluentPrediction = options.fluentprediction();
        useRelative = options.userelative();
        return ::mediapipe::OkStatus();
    }

    void SignLangPredictionCalculator::DoAfterInference()
    {
        framesSinceLastPrediction = 0;
        if (!usePoseLandmarks)
        {
            emptyFrames = 0;
        }
        if (!fluentPrediction)
        {
            framesWindow.clear();
        }
    }

    void SignLangPredictionCalculator::SetOutput(const std::string *str, ::mediapipe::CalculatorContext *cc)
    {
        cc->Outputs()
            .Tag(kTextOutputTag)
            .AddPacket(mediapipe::MakePacket<std::string>(*str)
                           .At(cc->InputTimestamp()));
    }

    ::mediapipe::Status SignLangPredictionCalculator::FillInputTensor(std::vector<std::vector<float>> localFrames)
    {
        int input = interpreter->inputs()[0];
        TfLiteIntArray *dims = interpreter->tensor(input)->dims;
        if (verboseLog)
        {
            LOG(INFO) << "Shape: {" << dims->data[0] << ", " << dims->data[1] << "}";
        }
        float *input_data_ptr = interpreter->typed_input_tensor<float>(0);
        RET_CHECK(input_data_ptr != nullptr);
        for (size_t i = 0; i < localFrames.size(); i++)
        {
            std::vector<float> frame = localFrames[i];
            for (size_t j = 0; j < frame.size(); j++)
            {
                *(input_data_ptr) = frame[j];
                input_data_ptr++;
            }
        }
        return ::mediapipe::OkStatus();
    }

    ::mediapipe::Status SignLangPredictionCalculator::UpdateFrames(CalculatorContext *cc)
    {
        std::vector<float> coordinates = {};

        if (usePoseLandmarks)
        {
            if (cc->Inputs().Tag(kPoseLandmarksTag).IsEmpty())
            {
                return ::mediapipe::OkStatus();
            }
            AddPoseLandmarks(coordinates, cc);
        }
        else
        {
            AddFaceDetectionsTo(coordinates, cc);
            if (coordinates.size() == 0)
            {                                        // No face detected.
                coordinates.push_back(defaultPoint); // 0 face_x
                coordinates.push_back(defaultPoint); // 0 face_y
            }
            AddMultiHandDetectionsTo(coordinates, cc);

            if (coordinates.size() < 44)
            { // No hands detected
                if (framesWindow.size() > minFramesForInference)
                {
                    emptyFrames++;
                }
                return ::mediapipe::OkStatus();
            }
        }
        int maxSize = use3D ? 128 : 86;
        maxSize = usePoseLandmarks ? 25 * 3 : maxSize;
        while (coordinates.size() < maxSize)
        {
            coordinates.push_back(defaultPoint);
        }
        if (coordinates.size() > maxSize)
        {
            LOG(ERROR) << "Coordinates size not equal " << maxSize << ". Actual size: " << coordinates.size();
            return ::mediapipe::OkStatus();
        }
        if (useRelative) {
            coordinates = GetCoordinatesRelative(coordinates);
        }

        while (framesWindow.size() >= framesWindowSize)
        {
            framesWindow.erase(framesWindow.begin());
        }

        // Put actual frame into array.
        framesWindow.push_back(coordinates);
        framesSinceLastPrediction++;
        return ::mediapipe::OkStatus();
    }

    bool SignLangPredictionCalculator::ShouldPredict()
    {
        // Minimum frames required for inference
        if (framesSinceLastPrediction < minFramesForInference)
        {
            return false;
        }
        if (usePoseLandmarks)
        {
            return true;
        }
        if (fluentPrediction)
        {
            return true;
        }
        // Long enough without hands to predict.
        if (emptyFrames >= thresholdFramesCount)
        {
            return true;
        }
        return false;
    }

    void SignLangPredictionCalculator::AddFaceDetectionsTo(std::vector<float> &coordinates, CalculatorContext *cc)
    {
        const std::vector<Detection> &faceDetections =
            cc->Inputs().Tag(kFaceDetectionsTag).Get<std::vector<Detection>>();

        if (!faceDetections.size())
        {
            return;
        }

        const Detection &face = faceDetections[0];
        LocationData locationData = face.location_data();
        int kpSize = locationData.relative_keypoints_size();

        if (!kpSize)
        {
            return;
        }

        auto keypoint = face.location_data().relative_keypoints(0);
        coordinates.push_back(keypoint.x());
        coordinates.push_back(keypoint.y());
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
                if (landmark.x() == 0 && landmark.y() == 0)
                {
                    continue;
                }
                coordinates.push_back(landmark.x());
                coordinates.push_back(landmark.y());
                if (use3D && landmark.has_z())
                {
                    coordinates.push_back(landmark.z());
                }
            }
        }
    }

    void SignLangPredictionCalculator::AddPoseLandmarks(std::vector<float> &coordinates, CalculatorContext *cc)
    {
        const NormalizedLandmarkList poseLandmarks = cc->Inputs().Tag(kPoseLandmarksTag).Get<NormalizedLandmarkList>();
        for (int i = 0; i < 25; ++i)
        {
            const NormalizedLandmark &landmark = poseLandmarks.landmark(i);
            coordinates.push_back(landmark.x());
            coordinates.push_back(landmark.y());
            coordinates.push_back(landmark.z());
        }
    }

    std::vector<float> SignLangPredictionCalculator::GetCoordinatesRelative(std::vector<float> coordinatesB) {
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

    REGISTER_CALCULATOR(SignLangPredictionCalculator);

} // namespace signlang

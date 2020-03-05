// Calculator für das Schreiben von Kopf- und Hand-Tracking-Punkten in eine CSV Datei.

#include <cstdio>
#include "mediapipe/framework/calculator_framework.h"
using namespace std;
namespace signlang
{
class DetectionsToCsvCalculator : public ::mediapipe::CalculatorBase
{
public:
    static ::mediapipe::Status GetContract(mediapipe::CalculatorContract *cc)
    {
        puts("GetContract called.");
        return ::mediapipe::OkStatus();
    }

    ::mediapipe::Status Open(mediapipe::CalculatorContext *cc) final
    {
        puts("Open called.");
        return ::mediapipe::OkStatus();
    }

    ::mediapipe::Status Process(mediapipe::CalculatorContext *cc) final
    {
        puts("Process called");
        return ::mediapipe::OkStatus();
    }

};

// ::mediapipe::REGISTER_CALCULATOR(DetectionsToCsvCalculator);
} // namespace signlang

// Calculator für das Schreiben von Kopf- und Hand-Tracking-Punkten in eine CSV Datei.

#include <cstdio>
#include "mediapipe/framework/calculator_framework.h"
using namespace std;
namespace signlang
{
class DetectionsToCSVCalculator : public ::mediapipe::CalculatorBase
{
public:
    static ::mediapipe::Status GetContract(mediapipe::CalculatorContract *cc)
    {
        puts("GetContract called.");
        const int tick_signal_index = cc->Inputs().NumEntries() - 1;
        // cc->Inputs().NumEntries() returns the number of input streams
        // for the PacketClonerCalculator
        for (int i = 0; i < tick_signal_index; ++i)
        {
            cc->Inputs().Index(i).SetAny();
        }
        cc->Inputs().Index(tick_signal_index).SetAny();
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

REGISTER_CALCULATOR(DetectionsToCSVCalculator);
} // namespace signlang

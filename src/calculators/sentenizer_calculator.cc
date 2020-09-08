#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/status.h"
#include <fstream>
#include <ctime>
#include <iomanip>
#include <iostream>

using namespace mediapipe;

namespace signlang
{
    constexpr char kEndOfSentence[] = "<eos>";
    constexpr int wordsToHold = 5;
    class SentenizerCalculator : public CalculatorBase
    {
    public:
        static ::mediapipe::Status GetContract(CalculatorContract *cc);
        ::mediapipe::Status Open(CalculatorContext *cc) override;
        ::mediapipe::Status Process(CalculatorContext *cc) override;
        ::mediapipe::Status Close(CalculatorContext *cc) override;

    private:
        std::string kPlaceholder = "Nothing yet...";
        std::vector<std::tuple<std::string, float>> lastNWords = {std::make_tuple(kPlaceholder, 1.0)};
        std::vector<std::string> sentence = {};
        std::fstream outputFile;
        std::string filename;
    };

    Status SentenizerCalculator::GetContract(CalculatorContract *cc)
    {
        cc->Inputs().Index(0).Set<std::tuple<std::string, float>>();
        cc->Outputs().Index(0).Set<std::string>();
        return OkStatus();
    }

    Status SentenizerCalculator::Open(CalculatorContext *cc)
    {
        time_t rawtime;
        struct tm *timeinfo;
        char buffer[80];

        time(&rawtime);
        timeinfo = localtime(&rawtime);

        strftime(buffer, sizeof(buffer), "%d-%m-%Y %H:%M:%S", timeinfo);
        std::string str(buffer);
        filename = "/home/signlang/" + str + ".csv";
        outputFile.open(filename, std::fstream::out);
        outputFile << "word;score" << std::endl;
        return OkStatus();
    }

    Status SentenizerCalculator::Process(CalculatorContext *cc)
    {
        auto nextWord = cc->Inputs().Index(0).Get<std::tuple<std::string, float>>();
        auto word = std::get<0>(nextWord);
        auto score = std::get<1>(nextWord);
        
        if (score == 0) {
            auto lastWord = std::get<0>(lastNWords[lastNWords.size() - 1]) + std::to_string(std::get<1>(lastNWords[lastNWords.size() - 1]));
            cc->Outputs().Index(0).AddPacket(MakePacket<std::string>(lastWord).At(cc->InputTimestamp()));
            return OkStatus();
        }

        lastNWords.push_back(nextWord);
        if (lastNWords.size() > wordsToHold)
        {
            lastNWords.erase(lastNWords.begin());
        }
        outputFile << word << ";" << score << std::endl;
        auto lastWord = std::get<0>(lastNWords[lastNWords.size() - 1]) + ": " + std::to_string(std::get<1>(lastNWords[lastNWords.size() - 1]));
        cc->Outputs().Index(0).AddPacket(MakePacket<std::string>(lastWord).At(cc->InputTimestamp()));
        return OkStatus();
    }

    Status SentenizerCalculator::Close(CalculatorContext *cc){
        outputFile.close();
        return OkStatus();
    }
    REGISTER_CALCULATOR(SentenizerCalculator);
} // namespace signlang
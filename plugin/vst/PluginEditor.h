#pragma once
// A plain editor: choose a model file, mix, bypass, high band, what the model
// is, the latency and why audio passes through (if it does). Runs on the
// message thread only; it talks to the processor through its control calls
// and parameters, never to the audio thread.

#include "PluginProcessor.h"

#include <juce_audio_processors/juce_audio_processors.h>

#include <memory>

class DenoiserEditor final : public juce::AudioProcessorEditor, private juce::ChangeListener {
public:
    explicit DenoiserEditor(DenoiserProcessor& owner);
    ~DenoiserEditor() override;

    void paint(juce::Graphics& g) override;
    void resized() override;

private:
    void changeListenerCallback(juce::ChangeBroadcaster*) override { refresh(); }
    void refresh();
    void chooseModel();

    DenoiserProcessor& processor_;

    juce::TextButton loadButton_{"Load model..."};
    juce::TextButton clearButton_{"Unload"};
    juce::Label pathLabel_, descriptionLabel_, latencyLabel_, messageLabel_;
    juce::Label mixLabel_{{}, "Mix"}, highBandLabel_{{}, "Above 8 kHz"};
    juce::Slider mixSlider_{juce::Slider::LinearHorizontal, juce::Slider::TextBoxRight};
    juce::ToggleButton bypassButton_{"Bypass"};
    juce::ComboBox highBandBox_;

    juce::AudioProcessorValueTreeState::SliderAttachment mixAttachment_;
    juce::AudioProcessorValueTreeState::ButtonAttachment bypassAttachment_;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment> highBandAttachment_;

    std::unique_ptr<juce::FileChooser> chooser_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(DenoiserEditor)
};

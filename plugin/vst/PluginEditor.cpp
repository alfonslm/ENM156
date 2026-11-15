#include "PluginEditor.h"

DenoiserEditor::DenoiserEditor(DenoiserProcessor& processor)
    : AudioProcessorEditor(processor),
      processor_(processor),
      mixAttachment_(processor.parameters(), DenoiserProcessor::kMixId, mixSlider_),
      bypassAttachment_(processor.parameters(), DenoiserProcessor::kBypassId, bypassButton_) {
    // The combo box must have its items before the attachment is made
    highBandBox_.addItem("Drop (model band only, up to ~7.5 kHz)", 1);
    highBandBox_.addItem("Pass through (unprocessed, noise stays)", 2);
    highBandAttachment_ = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(
        processor.parameters(), DenoiserProcessor::kHighBandId, highBandBox_);

    loadButton_.onClick = [this] { chooseModel(); };
    clearButton_.onClick = [this] { processor_.clearModel(); };

    for (auto* label : {&pathLabel_, &descriptionLabel_, &latencyLabel_, &messageLabel_}) {
        label->setJustificationType(juce::Justification::centredLeft);
        label->setMinimumHorizontalScale(0.7f);
    }
    messageLabel_.setColour(juce::Label::textColourId, juce::Colours::orange);
    messageLabel_.setJustificationType(juce::Justification::topLeft);
    mixSlider_.setTextValueSuffix(" %");

    for (juce::Component* c : std::initializer_list<juce::Component*>{
             &loadButton_, &clearButton_, &pathLabel_, &descriptionLabel_, &latencyLabel_, &messageLabel_,
             &mixLabel_, &mixSlider_, &bypassButton_, &highBandLabel_, &highBandBox_})
        addAndMakeVisible(c);

    processor_.addChangeListener(this);
    setSize(560, 280);
    refresh();
}

DenoiserEditor::~DenoiserEditor() {
    processor_.removeChangeListener(this);
}

void DenoiserEditor::chooseModel() {
    const juce::String current = processor_.getStatus().modelPath;
    const juce::File start = juce::File::isAbsolutePath(current) ? juce::File(current).getParentDirectory()
                                                                 : juce::File::getSpecialLocation(juce::File::userHomeDirectory);
    chooser_ = std::make_unique<juce::FileChooser>("Choose a denoiser model file", start, "*.bin");
    chooser_->launchAsync(juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
                          [this](const juce::FileChooser& fc) {
                              const juce::File file = fc.getResult();
                              if (file != juce::File())
                                  processor_.loadModelFile(file);  // reports back through the change broadcaster
                          });
}

void DenoiserEditor::refresh() {
    const DenoiserProcessor::Status s = processor_.getStatus();
    pathLabel_.setText(s.modelPath.isEmpty() ? juce::String("Model: none") : "Model: " + s.modelPath,
                       juce::dontSendNotification);
    pathLabel_.setTooltip(s.modelPath);
    descriptionLabel_.setText(s.description.isEmpty() ? juce::String("-") : s.description,
                              juce::dontSendNotification);
    juce::String latency;
    if (!s.prepared)
        latency = "Latency: known once the host starts processing";
    else
        latency = "Latency: " + juce::String(s.latencySamples) + " samples (" +
                  juce::String(1000.0 * s.latencySamples / s.sampleRate, 2) + " ms at " +
                  juce::String(s.sampleRate / 1000.0, 1) + " kHz)" + (s.denoising ? "" : ", passing audio through");
    latencyLabel_.setText(latency, juce::dontSendNotification);
    messageLabel_.setText(s.message, juce::dontSendNotification);
    clearButton_.setEnabled(s.modelPath.isNotEmpty());
}

void DenoiserEditor::paint(juce::Graphics& g) {
    g.fillAll(getLookAndFeel().findColour(juce::ResizableWindow::backgroundColourId));
}

void DenoiserEditor::resized() {
    auto area = getLocalBounds().reduced(12);
    auto row = [&area](int height) {
        auto r = area.removeFromTop(height);
        area.removeFromTop(6);
        return r;
    };
    {
        auto r = row(26);
        loadButton_.setBounds(r.removeFromLeft(120));
        r.removeFromLeft(6);
        clearButton_.setBounds(r.removeFromLeft(70));
        r.removeFromLeft(8);
        pathLabel_.setBounds(r);
    }
    descriptionLabel_.setBounds(row(22));
    latencyLabel_.setBounds(row(22));
    {
        auto r = row(26);
        mixLabel_.setBounds(r.removeFromLeft(90));
        bypassButton_.setBounds(r.removeFromRight(90));
        mixSlider_.setBounds(r);
    }
    {
        auto r = row(26);
        highBandLabel_.setBounds(r.removeFromLeft(90));
        highBandBox_.setBounds(r);
    }
    messageLabel_.setBounds(area);
}

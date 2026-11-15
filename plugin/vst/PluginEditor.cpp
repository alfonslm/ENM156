#include "PluginEditor.h"

DenoiserEditor::DenoiserEditor(DenoiserProcessor& owner)
    : AudioProcessorEditor(owner),
      processor_(owner),
      mixAttachment_(owner.parameters(), DenoiserProcessor::kMixId, mixSlider_),
      bypassAttachment_(owner.parameters(), DenoiserProcessor::kBypassId, bypassButton_) {
    // The combo box must have its items before the attachment is made
    highBandBox_.addItem("Drop (model band only, up to ~7.5 kHz)", 1);
    highBandBox_.addItem("Pass through (unprocessed, noise stays)", 2);
    highBandAttachment_ = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(
        processor_.parameters(), DenoiserProcessor::kHighBandId, highBandBox_);
    araHighBandBox_.addItem("Drop (model band only, up to ~7.5 kHz) - all clips", 1);
    araHighBandBox_.addItem("Pass through (unprocessed, noise stays) - all clips", 2);
#if JucePlugin_Enable_ARA
    araHighBandBox_.onChange = [this] {
        if (auto* dc = processor_.getAraDocumentController())
            dc->setHighBand(araHighBandBox_.getSelectedId() == 2 ? denoiser::HighBand::PassThrough
                                                                 : denoiser::HighBand::Drop);
    };
#endif
    araSources_.setMultiLine(true);
    araSources_.setReadOnly(true);
    araSources_.setScrollbarsShown(true);
    araSources_.setCaretVisible(false);

    loadButton_.onClick = [this] { chooseModel(); };
    clearButton_.onClick = [this] {
#if JucePlugin_Enable_ARA
        if (auto* dc = processor_.getAraDocumentController()) {
            dc->setModelFile({});
            refresh();
            return;
        }
#endif
        processor_.clearModel();
    };

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
    addChildComponent(araHighBandBox_);
    addChildComponent(araSources_);

    processor_.addChangeListener(this);
    setSize(560, araMode() ? 420 : 280);
    refresh();
    startTimerHz(5);  // ARA analysis progress (and binding after the window opened)
}

bool DenoiserEditor::araMode() const {
#if JucePlugin_Enable_ARA
    return processor_.getAraDocumentController() != nullptr;
#else
    return false;
#endif
}

DenoiserEditor::~DenoiserEditor() {
    stopTimer();
    processor_.removeChangeListener(this);
}

void DenoiserEditor::chooseModel() {
    juce::String current = processor_.getStatus().modelPath;
#if JucePlugin_Enable_ARA
    if (auto* dc = processor_.getAraDocumentController())
        current = dc->getModelStatus().path;
#endif
    const juce::File start = juce::File::isAbsolutePath(current) ? juce::File(current).getParentDirectory()
                                                                 : juce::File::getSpecialLocation(juce::File::userHomeDirectory);
    chooser_ = std::make_unique<juce::FileChooser>("Choose a denoiser model file", start, "*.bin");
    chooser_->launchAsync(juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
                          [this](const juce::FileChooser& fc) {
                              const juce::File file = fc.getResult();
                              if (file == juce::File())
                                  return;
#if JucePlugin_Enable_ARA
                              if (auto* dc = processor_.getAraDocumentController()) {
                                  dc->setModelFile(file.getFullPathName());  // the document's model
                                  refresh();
                                  return;
                              }
#endif
                              processor_.loadModelFile(file);  // reports back through the change broadcaster
                          });
}

void DenoiserEditor::refresh() {
    const bool ara = araMode();
    highBandBox_.setVisible(!ara);
    araHighBandBox_.setVisible(ara);
    araSources_.setVisible(ara);
    if (ara) {
        refreshAra();
        return;
    }
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

void DenoiserEditor::refreshAra() {
#if JucePlugin_Enable_ARA
    auto* dc = processor_.getAraDocumentController();
    if (dc == nullptr)
        return;
    const auto model = dc->getModelStatus();
    pathLabel_.setText(model.path.isEmpty() ? juce::String("Model (all clips): none")
                                            : "Model (all clips): " + model.path,
                       juce::dontSendNotification);
    pathLabel_.setTooltip(model.path);
    descriptionLabel_.setText(model.description.isEmpty() ? juce::String("-") : model.description,
                              juce::dontSendNotification);
    araHighBandBox_.setSelectedId(model.highBand == denoiser::HighBand::PassThrough ? 2 : 1,
                                  juce::dontSendNotification);
    clearButton_.setEnabled(model.path.isNotEmpty());

    const auto sources = dc->getSourceStatuses();
    int done = 0, busy = 0;
    juce::String list;
    for (const auto& s : sources) {
        if (s.state == DenoiserDocumentController::SourceState::Done)
            ++done;
        if (s.state == DenoiserDocumentController::SourceState::Reading ||
            s.state == DenoiserDocumentController::SourceState::Denoising)
            ++busy;
        list << s.name << " (" << juce::String(s.sampleRate / 1000.0, 1) << " kHz, " << s.channels << " ch, "
             << juce::String(s.sampleRate > 0.0 ? double(s.length) / s.sampleRate : 0.0, 1) << " s): "
             << DenoiserDocumentController::describe(s.state);
        if (s.state == DenoiserDocumentController::SourceState::Denoising ||
            s.state == DenoiserDocumentController::SourceState::Reading)
            list << " " << juce::roundToInt(100.0f * s.progress) << " %";
        if (s.message.isNotEmpty())
            list << " - " << s.message;
        list << "\n";
    }
    latencyLabel_.setText("ARA: whole clips, no latency. " + juce::String(int(sources.size())) + " source(s), " +
                              juce::String(done) + " denoised" +
                              (busy > 0 ? ", " + juce::String(busy) + " in progress" : juce::String()),
                          juce::dontSendNotification);
    messageLabel_.setText(model.message, juce::dontSendNotification);
    if (araSources_.getText() != list)
        araSources_.setText(list, false);
#endif
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
        araHighBandBox_.setBounds(r);
    }
    if (araMode()) {
        messageLabel_.setBounds(area.removeFromTop(40));
        area.removeFromTop(6);
        araSources_.setBounds(area);
    } else {
        messageLabel_.setBounds(area);
    }
}

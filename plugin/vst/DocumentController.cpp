// ARA 2 entry point. Compiled only when the plugin is built with ARA
// (DENOISER_ENABLE_ARA). The document controller does nothing yet: offline
// analysis and playback of whole audio sources is a later issue.
#include <juce_audio_processors/juce_audio_processors.h>

#if JucePlugin_Enable_ARA

class DenoiserDocumentController final : public juce::ARADocumentControllerSpecialisation {
public:
    using juce::ARADocumentControllerSpecialisation::ARADocumentControllerSpecialisation;

protected:
    // Nothing to store yet; must succeed so hosts can save their documents
    bool doRestoreObjectsFromStream(juce::ARAInputStream&, const juce::ARARestoreObjectsFilter*) override {
        return true;
    }
    bool doStoreObjectsToStream(juce::ARAOutputStream&, const juce::ARAStoreObjectsFilter*) override {
        return true;
    }
};

const ARA::ARAFactory* JUCE_CALLTYPE createARAFactory() {
    return juce::ARADocumentControllerSpecialisation::createARAFactory<DenoiserDocumentController>();
}

#endif

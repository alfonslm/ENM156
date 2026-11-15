# The JUCE plugin: VST3, optionally ARA 2. Built only with DENOISER_BUILD_PLUGIN=ON.
# Add new plugin sources to target_sources() below; DSP belongs in core/.
juce_add_plugin(DenoiserPlugin
    PRODUCT_NAME "ENM156 Denoiser"
    COMPANY_NAME "ENM156"
    BUNDLE_ID org.enm156.denoiser
    PLUGIN_MANUFACTURER_CODE Enms   # four characters, at least one upper-case
    PLUGIN_CODE Dnz1                # four characters, exactly one upper-case
    FORMATS VST3
    VST3_CATEGORIES Fx Restoration
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE
    COPY_PLUGIN_AFTER_BUILD FALSE
    NEEDS_WEB_BROWSER FALSE
    NEEDS_CURL FALSE
    # ARA_FACTORY_ID and ARA_DOCUMENT_ARCHIVE_ID keep JUCE's defaults, made from
    # BUNDLE_ID: "org.enm156.denoiser.arafactory.<version>" and
    # "org.enm156.denoiser.aradocumentarchive.1" (bump the latter when the
    # stored document format changes)
    IS_ARA_EFFECT ${DENOISER_ENABLE_ARA})

target_sources(DenoiserPlugin
    PRIVATE
        PluginProcessor.cpp
        PluginEditor.cpp
        RealtimeEngine.cpp
        DocumentController.cpp)

target_compile_definitions(DenoiserPlugin
    PUBLIC
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        JUCE_VST3_CAN_REPLACE_VST2=0)

target_link_libraries(DenoiserPlugin
    PRIVATE
        denoiser::core
        juce::juce_audio_processors
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags)

# A minimal headless host that loads the built VST3 and checks the real-time
# denoiser through the plugin interface (ctest: plugin_vst3)
if(DENOISER_BUILD_TESTS)
    juce_add_console_app(denoiser_plugin_host_test
        PRODUCT_NAME "denoiser_plugin_host_test"
        NEEDS_WEB_BROWSER FALSE
        NEEDS_CURL FALSE)
    target_sources(denoiser_plugin_host_test PRIVATE HostTest.cpp)
    target_compile_definitions(denoiser_plugin_host_test
        PRIVATE
            JUCE_PLUGINHOST_VST3=1
            JUCE_PLUGINHOST_ARA=$<BOOL:${DENOISER_ENABLE_ARA}>
            DENOISER_EXPECT_ARA=$<BOOL:${DENOISER_ENABLE_ARA}>
            JUCE_WEB_BROWSER=0
            JUCE_USE_CURL=0)
    target_link_libraries(denoiser_plugin_host_test
        PRIVATE
            denoiser::core
            juce::juce_audio_processors_headless
            juce::juce_events
        PUBLIC
            juce::juce_recommended_config_flags
            juce::juce_recommended_warning_flags)
    # Build the plugin whenever the test is built
    add_dependencies(denoiser_plugin_host_test DenoiserPlugin_VST3)

    get_target_property(denoiser_vst3_bundle DenoiserPlugin_VST3 JUCE_PLUGIN_ARTEFACT_FILE)
    add_test(NAME plugin_vst3
        COMMAND denoiser_plugin_host_test "${denoiser_vst3_bundle}" "${PROJECT_SOURCE_DIR}/tests/golden")
endif()

# denoiser_core: DSP + inference, plain C++17. No JUCE, no host code.
# Add new sources to the list below; public headers go in include/denoiser/.
add_library(denoiser_core STATIC
    src/core.cpp
    src/fft.cpp
    src/resample.cpp src/host_rate.cpp)
add_library(denoiser::core ALIAS denoiser_core)

target_include_directories(denoiser_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(denoiser_core PUBLIC cxx_std_17)
target_compile_definitions(denoiser_core PRIVATE DENOISER_VERSION="${PROJECT_VERSION}")
# KISS FFT is an implementation detail: callers use denoiser/fft.h
target_link_libraries(denoiser_core PRIVATE denoiser_kissfft)

if(MSVC)
    target_compile_options(denoiser_core PRIVATE /W4)
else()
    target_compile_options(denoiser_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Model/vectors file reader and the STFT front/back end (issue #5)
target_sources(denoiser_core PRIVATE src/json.cpp src/tensor_file.cpp src/model_file.cpp src/features.cpp src/stream.cpp src/offline.cpp)

# Incremental TCN inference (issue #7)
target_sources(denoiser_core PRIVATE src/tcn.cpp)

# MLP and SNN inference, the shared Model and the per-channel denoiser (issue #6)
target_sources(denoiser_core PRIVATE src/networks.cpp src/denoiser.cpp)

# Whole clips at other sample rates, for the ARA analysis (issue #10)
target_sources(denoiser_core PRIVATE src/offline_rate.cpp)

# denoiser_core: DSP + inference, plain C++17. No JUCE, no host code.
# Add new sources to the list below; public headers go in include/denoiser/.
add_library(denoiser_core STATIC
    src/core.cpp
    src/fft.cpp)
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

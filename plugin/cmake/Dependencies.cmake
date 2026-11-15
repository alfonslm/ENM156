# Third-party dependencies, fetched by git at pinned versions.
#
# Every one can be pointed at a local checkout instead (offline builds, or to
# share one checkout between build folders):
#   -DFETCHCONTENT_SOURCE_DIR_KISSFFT=/path/to/kissfft
#   -DFETCHCONTENT_SOURCE_DIR_JUCE=/path/to/JUCE
#   -DFETCHCONTENT_SOURCE_DIR_ARA_SDK=/path/to/ARA_SDK   (cloned with --recursive)
# A local checkout must be at the same version as below.
include(FetchContent)

# --- KISS FFT (BSD-3-Clause): real FFTs of any even size, e.g. 1280 and 128 ---
# Tag 131.2.0 = commit 7bce4153c6bc8aba2db0e889e576f9d00505cbe1
FetchContent_Declare(kissfft
    GIT_REPOSITORY https://github.com/mborgerding/kissfft.git
    GIT_TAG        131.2.0
    GIT_SHALLOW    TRUE
    # Only fetch: its own CMakeLists adds tests and tools we don't want, so we
    # build the two source files we need ourselves (below)
    SOURCE_SUBDIR  do-not-add-subdirectory)
FetchContent_MakeAvailable(kissfft)

add_library(denoiser_kissfft STATIC
    ${kissfft_SOURCE_DIR}/kiss_fft.c
    ${kissfft_SOURCE_DIR}/kiss_fftr.c)
target_include_directories(denoiser_kissfft SYSTEM PUBLIC ${kissfft_SOURCE_DIR})
# Single precision, like the PyTorch model; no OpenMP, no SIMD
target_compile_definitions(denoiser_kissfft PUBLIC kiss_fft_scalar=float)
if(NOT MSVC)
    target_link_libraries(denoiser_kissfft PRIVATE m)
endif()

if(DENOISER_BUILD_PLUGIN)
    # --- JUCE (AGPL-3.0-only OR commercial) ---
    # Tag 9.0.5 = commit d60b7a78030661574a615b586d9325ba7a7fa7dd
    FetchContent_Declare(juce
        GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
        GIT_TAG        9.0.5
        GIT_SHALLOW    TRUE
        GIT_PROGRESS   TRUE)

    # --- ARA SDK (Apache-2.0), the version JUCE 9.0.5's docs/ARA.md names ---
    # Tag releases/2.3.0 = commit 0005c532bb03eb210e7bb490329fc6d7ba1792c7,
    # with its submodules ARA_API, ARA_Library and ARA_Examples
    FetchContent_Declare(ara_sdk
        GIT_REPOSITORY https://github.com/Celemony/ARA_SDK.git
        GIT_TAG        releases/2.3.0
        GIT_SHALLOW    TRUE
        GIT_PROGRESS   TRUE
        # Only fetch: JUCE compiles the parts of it that it needs
        SOURCE_SUBDIR  do-not-add-subdirectory)

    FetchContent_MakeAvailable(juce ara_sdk)

    if(DENOISER_ENABLE_ARA)
        # Must come before any plugin target is added
        juce_set_ara_sdk_path(${ara_sdk_SOURCE_DIR})
    endif()
endif()

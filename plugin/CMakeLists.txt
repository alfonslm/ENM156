# Denoiser plugin: C++ core, command-line tool, tests and (optionally) the
# JUCE VST3 + ARA plugin. See README.md in this folder.
#
#   cmake -S plugin -B plugin/build                              # core + cli + tests, no JUCE
#   cmake -S plugin -B plugin/build -DDENOISER_BUILD_PLUGIN=ON   # ... and the VST3 plugin
cmake_minimum_required(VERSION 3.22)  # what JUCE 9 asks for

project(ENM156Denoiser VERSION 0.1.0 LANGUAGES C CXX)

option(DENOISER_BUILD_PLUGIN "Fetch JUCE and the ARA SDK and build the VST3 plugin" OFF)
option(DENOISER_ENABLE_ARA "Build the plugin as an ARA 2 plugin (needs DENOISER_BUILD_PLUGIN)" ON)
option(DENOISER_BUILD_CLI "Build the command-line tool" ON)
option(DENOISER_BUILD_TESTS "Build the core tests (run them with ctest)" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# The core is linked into the plugin, which is a shared module
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Third-party code, fetched with git at pinned versions
include(cmake/Dependencies.cmake)

add_subdirectory(core)

if(DENOISER_BUILD_CLI)
    add_subdirectory(cli)
endif()

if(DENOISER_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(DENOISER_BUILD_PLUGIN)
    add_subdirectory(vst)
endif()

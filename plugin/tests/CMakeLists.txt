# Core tests, run by ctest. Each test is one small executable using check.h;
# to add one, write test_<name>.cpp and add <name> to this list.
# (golden/ holds test vectors written by the Python export script.)
set(DENOISER_TESTS
    core
    fft
    resample host_rate
    model_file stft_golden stft_alloc
    networks denoiser_golden networks_alloc)

foreach(name IN LISTS DENOISER_TESTS)
    add_executable(test_${name} test_${name}.cpp)
    target_link_libraries(test_${name} PRIVATE denoiser::core)
    add_test(NAME ${name} COMMAND test_${name})
endforeach()

# Tests that read golden/ get its path compiled in (or take it as their first argument)
foreach(name IN LISTS DENOISER_TESTS)
    target_compile_definitions(test_${name} PRIVATE DENOISER_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/golden")
endforeach()

if(TARGET denoiser_cli)
    add_test(NAME cli_version COMMAND denoiser_cli --version)
    set_tests_properties(cli_version PROPERTIES PASS_REGULAR_EXPRESSION "^denoise [0-9]")
endif()

# The command-line tool run as a program on golden inputs, and its WAV code (issue #6)
if(TARGET denoiser_cli)
    add_executable(test_cli test_cli.cpp ${PROJECT_SOURCE_DIR}/cli/wav.cpp)
    target_link_libraries(test_cli PRIVATE denoiser::core)
    add_test(NAME cli COMMAND test_cli $<TARGET_FILE:denoiser_cli> ${CMAKE_CURRENT_SOURCE_DIR}/golden
             ${CMAKE_CURRENT_BINARY_DIR}/cli_work)
endif()

# Speed of the networks and the per-channel denoiser on model files (not a
# test; run by hand, see README.md "Inference")
add_executable(bench_networks bench_networks.cpp)
target_link_libraries(bench_networks PRIVATE denoiser::core)

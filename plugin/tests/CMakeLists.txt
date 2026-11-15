# Core tests, run by ctest. Each test is one small executable using check.h;
# to add one, write test_<name>.cpp and add <name> to this list.
# (golden/ holds test vectors written by the Python export script.)
set(DENOISER_TESTS
    core
    fft
    resample host_rate)

foreach(name IN LISTS DENOISER_TESTS)
    add_executable(test_${name} test_${name}.cpp)
    target_link_libraries(test_${name} PRIVATE denoiser::core)
    add_test(NAME ${name} COMMAND test_${name})
endforeach()

if(TARGET denoiser_cli)
    add_test(NAME cli_version COMMAND denoiser_cli --version)
    set_tests_properties(cli_version PROPERTIES PASS_REGULAR_EXPRESSION "^denoise [0-9]")
endif()

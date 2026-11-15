# The command-line denoiser, built as `denoise`. Add new sources to this list.
add_executable(denoiser_cli
    main.cpp)
set_target_properties(denoiser_cli PROPERTIES OUTPUT_NAME denoise)
target_link_libraries(denoiser_cli PRIVATE denoiser::core)

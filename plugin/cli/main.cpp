// Command-line denoiser built on the core. Placeholder until the core can
// load a model: the planned usage is
//   denoise model.bin in.wav out.wav      (16 kHz mono, see ../README.md)
#include "denoiser/core.h"

#include <cstdio>
#include <cstring>

int main(int argc, char** argv) {
    if (argc == 2 && (std::strcmp(argv[1], "--version") == 0 || std::strcmp(argv[1], "-V") == 0)) {
        std::printf("denoise %s\n", denoiser::version());
        return 0;
    }
    std::fprintf(stderr,
                 "usage: denoise model.bin in.wav out.wav\n"
                 "       denoise --version\n"
                 "denoising is not implemented yet (core %s)\n",
                 denoiser::version());
    return 2;
}

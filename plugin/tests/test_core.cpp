// The placeholder core API: replace these checks as the real API arrives.
#include "check.h"
#include "denoiser/core.h"

#include <cstring>
#include <vector>

int main() {
    CHECK(std::strlen(denoiser::version()) > 0);

    std::vector<float> in = {0.0f, 0.5f, -1.0f, 0.25f};
    std::vector<float> out(in.size(), 9.0f);
    denoiser::passThrough(in.data(), out.data(), in.size());
    CHECK(out == in);

    denoiser::passThrough(in.data(), in.data(), in.size());  // in place
    CHECK(out == in);

    return check::finish();
}

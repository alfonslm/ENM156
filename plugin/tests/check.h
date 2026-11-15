#pragma once
// A minimal test helper: CHECK(cond) prints the failing line and counts it,
// and a test's main() ends with `return check::finish();`. Works in Release
// builds, unlike assert().

#include <cstdio>

namespace check {

inline int& failures() {
    static int count = 0;
    return count;
}

inline int finish() {
    if (failures() == 0) {
        std::printf("all checks passed\n");
        return 0;
    }
    std::printf("%d check(s) failed\n", failures());
    return 1;
}

}  // namespace check

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            ++check::failures();                                           \
        }                                                                  \
    } while (0)

// CHECK that |a - b| <= tol, printing both values if not
#define CHECK_NEAR(a, b, tol)                                                          \
    do {                                                                               \
        const double check_a_ = (a), check_b_ = (b);                                   \
        if (!(check_a_ - check_b_ <= (tol) && check_b_ - check_a_ <= (tol))) {          \
            std::printf("%s:%d: CHECK_NEAR failed: %s = %.9g, %s = %.9g, tol %g\n",    \
                        __FILE__, __LINE__, #a, check_a_, #b, check_b_, (double)(tol)); \
            ++check::failures();                                                       \
        }                                                                              \
    } while (0)

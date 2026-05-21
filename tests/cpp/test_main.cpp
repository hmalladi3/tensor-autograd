#include "test_main.h"

#include <cstdio>
#include <exception>

namespace test {

int run_all() {
    int passed = 0, failed = 0;
    for (auto& tc : registry()) {
        std::vector<Failure> failures;
        try {
            tc.fn(failures);
        } catch (const std::exception& e) {
            failures.push_back({"<exception>", 0, std::string("uncaught: ") + e.what()});
        } catch (...) {
            failures.push_back({"<exception>", 0, "uncaught: non-std exception"});
        }
        if (failures.empty()) {
            std::printf("  ok    %s.%s\n", tc.suite, tc.name);
            ++passed;
        } else {
            std::printf("  FAIL  %s.%s\n", tc.suite, tc.name);
            for (auto& f : failures) {
                std::printf("        %s:%d  %s\n", f.file.c_str(), f.line, f.text.c_str());
            }
            ++failed;
        }
    }
    std::printf("\n%d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}

}  // namespace test

int main() { return test::run_all(); }

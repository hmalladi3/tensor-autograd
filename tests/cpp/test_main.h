/*
 * test_main.h — hand-rolled, zero-dependency unit test harness.
 *
 * Define tests with TEST(suite, name) { ... }. They self-register.
 * Use CHECK(expr) for non-fatal asserts and REQUIRE(expr) to abort the test.
 * CHECK_EQ(a, b), CHECK_NEAR(a, b, tol), CHECK_THROWS(expr) cover the common cases.
 *
 * Build one test executable per source file (see CMakeLists.txt in this dir).
 * Each executable links against test_main.cpp, which supplies `main`.
 */

#ifndef TEST_MAIN_H
#define TEST_MAIN_H

#include <cmath>
#include <string>
#include <vector>

namespace test {

struct Failure {
    std::string file;
    int         line;
    std::string text;
};

struct TestCase {
    const char* suite;
    const char* name;
    void      (*fn)(std::vector<Failure>&);
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> r;
    return r;
}

struct AutoRegister {
    AutoRegister(const char* suite, const char* name, void (*fn)(std::vector<Failure>&)) {
        registry().push_back({suite, name, fn});
    }
};

int run_all();

}  // namespace test

#define TEST(suite, name)                                                                       \
    static void test_##suite##_##name(std::vector<test::Failure>& _failures);                   \
    static test::AutoRegister _reg_##suite##_##name(#suite, #name, test_##suite##_##name);      \
    static void test_##suite##_##name([[maybe_unused]] std::vector<test::Failure>& _failures)

#define CHECK(expr)                                                                             \
    do {                                                                                        \
        if (!(expr)) _failures.push_back({__FILE__, __LINE__, "CHECK(" #expr ")"});             \
    } while (0)

#define REQUIRE(expr)                                                                           \
    do {                                                                                        \
        if (!(expr)) {                                                                          \
            _failures.push_back({__FILE__, __LINE__, "REQUIRE(" #expr ")"});                    \
            return;                                                                             \
        }                                                                                       \
    } while (0)

#define CHECK_EQ(a, b)                                                                          \
    do {                                                                                        \
        auto _va = (a);                                                                         \
        auto _vb = (b);                                                                         \
        if (!(_va == _vb))                                                                      \
            _failures.push_back({__FILE__, __LINE__, "CHECK_EQ(" #a ", " #b ")"});              \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                                   \
    do {                                                                                        \
        auto _va = (a); auto _vb = (b);                                                         \
        if (std::fabs(_va - _vb) > (tol))                                                       \
            _failures.push_back({__FILE__, __LINE__, "CHECK_NEAR(" #a ", " #b ")"});            \
    } while (0)

#define CHECK_THROWS(expr)                                                                      \
    do {                                                                                        \
        bool _threw = false;                                                                    \
        try { (void)(expr); } catch (...) { _threw = true; }                                    \
        if (!_threw) _failures.push_back({__FILE__, __LINE__, "CHECK_THROWS(" #expr ")"});      \
    } while (0)

#endif  // TEST_MAIN_H

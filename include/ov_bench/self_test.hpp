#pragma once
#include <string>

namespace ovbench {
int run_random_accuracy_core_tests();
int run_gather_guard_test(const std::string& plugin);
}
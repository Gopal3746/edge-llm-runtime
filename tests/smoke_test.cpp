#include <cstdlib>
#include <iostream>

#include "edge_llm/version.hpp"

int main() {
    if (edge_llm::version.empty()) {
        std::cerr << "Runtime version must not be empty\n";
        return EXIT_FAILURE;
    }

    std::cout << "Smoke test passed\n";
    return EXIT_SUCCESS;
}
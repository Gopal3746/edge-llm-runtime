#include <iostream>

#include "edge_llm/version.hpp"

int main() {
    std::cout
        << "Edge LLM Runtime "
        << edge_llm::version
        << '\n';

    return 0;
}
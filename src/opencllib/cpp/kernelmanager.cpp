#include "kernelmanager.hpp"

KernelManager::KernelSourceInfo KernelManager::getKernelSource(operation_types op_type) const {
    auto location = lookup_table.find(op_type);
    if (location == lookup_table.end()) {
        throw std::runtime_error("Kernel binding name not found");
    }

    // Check if we already have the kernel source loaded
    auto existing_source = kernel_sources.find(op_type);
    if (existing_source == kernel_sources.end()) {
        // Only load the file if we haven't already
        std::ifstream file(location->second);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open kernel file: " + location->second);
        }

        std::stringstream buffer;
        buffer << file.rdbuf();
        file.close();

        // Store the source
        kernel_sources.insert({op_type, buffer.str()});
    }

    // Update the pointer to the persistent string data
    current_source = kernel_sources.at(op_type).c_str();
    source_array[0] = current_source;

    KernelSourceInfo info;
    info.source = source_array;
    info.length = kernel_sources.at(op_type).length();
    return info;
}
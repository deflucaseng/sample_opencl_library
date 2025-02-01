#ifndef KERNELMANAGER_HPP
#define KERNELMANAGER_HPP

#include "operationtypes.hpp"
#include <unordered_map>
#include <string>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <iostream>

class KernelManager
{

public:
	struct KernelSourceInfo {
		const char** source;
		size_t length;
	};

    KernelSourceInfo getKernelSource(operation_types op_type) const;

private:
	std::unordered_map<operation_types, std::string> lookup_table = {

		//	Name used in Binding						File Location

		{operation_types::ADD, "src/opencllib/cpp/kernels/add.cl"},
		{operation_types::DIV, "src/opencllib/cpp/kernels/div.cl"},
		{operation_types::MUL, "src/opencllib/cpp/kernels/mul.cl"},
		{operation_types::SUB, "src/opencllib/cpp/kernels/sub.cl"}
	};
	mutable std::unordered_map<operation_types, std::string> kernel_sources;
	mutable const char *current_source;
	mutable const char *source_array[1]; // Array of size 1 for OpenCL
};

#endif

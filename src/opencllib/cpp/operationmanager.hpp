#ifndef OPERATIONMANAGER_HPP
#define OPERATIONMANAGER_HPP
#include <vector>
#include "pch.hpp"
class OperationManager{

	public:
		
		enum class DeviceType{
			CPU_DEVICE,
			GPU_DEVICE
		};

		OperationManager(DeviceType device_type); // Determine if we will use a CPU/GPU for calculation
		~OperationManager();

		float* element_wise_operation(operation_types op_type, float *lhs, float* rhs, int height, int width);


	private:
		KernelManager kernel_manager;

		cl_platform_id platform;
		cl_device_id device;
		cl_uint num_platforms, num_devices;
		cl_context context;
		cl_command_queue queue;


};


#endif
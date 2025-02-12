#include "operationmanager.hpp"
std::string getOpenCLErrorString(cl_int error) {
    switch (error) {
        case CL_SUCCESS: return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_MEM_COPY_OVERLAP: return "CL_MEM_COPY_OVERLAP";
        case CL_IMAGE_FORMAT_MISMATCH: return "CL_IMAGE_FORMAT_MISMATCH";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
        case CL_MAP_FAILURE: return "CL_MAP_FAILURE";
        case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE_TYPE: return "CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_PLATFORM: return "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES: return "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_HOST_PTR: return "CL_INVALID_HOST_PTR";
        case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case CL_INVALID_IMAGE_SIZE: return "CL_INVALID_IMAGE_SIZE";
        case CL_INVALID_SAMPLER: return "CL_INVALID_SAMPLER";
        case CL_INVALID_BINARY: return "CL_INVALID_BINARY";
        case CL_INVALID_BUILD_OPTIONS: return "CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION: return "CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION: return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET: return "CL_INVALID_GLOBAL_OFFSET";
        case CL_INVALID_EVENT_WAIT_LIST: return "CL_INVALID_EVENT_WAIT_LIST";
        case CL_INVALID_EVENT: return "CL_INVALID_EVENT";
        case CL_INVALID_OPERATION: return "CL_INVALID_OPERATION";
        case CL_INVALID_GL_OBJECT: return "CL_INVALID_GL_OBJECT";
        case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_MIP_LEVEL: return "CL_INVALID_MIP_LEVEL";
        case CL_INVALID_GLOBAL_WORK_SIZE: return "CL_INVALID_GLOBAL_WORK_SIZE";
        default: return "UNKNOWN ERROR: " + std::to_string(error);
    }
}

OperationManager::OperationManager(DeviceType device_type){
	clGetPlatformIDs(1, &platform, &num_platforms);

	switch (device_type)
	{
	case (DeviceType::CPU_DEVICE):
		clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, &num_devices);

	case (DeviceType::GPU_DEVICE):
		clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &num_devices);
	}

	// Step 2: Create Context and Command Queue
	context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
	queue = clCreateCommandQueue(context, device, 0, NULL);
}

OperationManager::~OperationManager()
{
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
}

float* OperationManager::element_wise_operation(operation_types op_type, float *lhs, float* rhs, int height, int width){
	cl_int err;


    auto kernel_info = kernel_manager.getKernelSource(op_type);

	cl_program program = clCreateProgramWithSource(context, 1, kernel_info.source, NULL, &err);
	if (err != CL_SUCCESS)
	{
		std::cout << getOpenCLErrorString(err) << std::endl;
		throw std::runtime_error("Failed to create program");
	}

		err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	
	
	if (err != CL_SUCCESS)
	{
		// Get build log for debugging
		size_t log_size;
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		std::vector<char> build_log(log_size);
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), NULL);
		clReleaseProgram(program);
		throw std::runtime_error("Failed to build program: " + std::string(build_log.data()));
	}


	cl_kernel kernel = clCreateKernel(program, "op_kernel", &err);
	if (err != CL_SUCCESS) {
		clReleaseProgram(program);
		throw std::runtime_error("Failed to create kernel: " + getOpenCLErrorString(err));
	}


	const int matrix_size = height * width * sizeof(float);

	float *matrix_result = (float *)malloc(matrix_size);
	if (!matrix_result)
	{
		throw std::bad_alloc();
	}


	cl_mem lhs_buffer = nullptr;
	cl_mem rhs_buffer = nullptr;
	cl_mem result_buffer = nullptr;

	lhs_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
								matrix_size, lhs, &err);
	rhs_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
								matrix_size, rhs, &err);
	result_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
								matrix_size, NULL, &err);

		
	if (err != CL_SUCCESS)
	{
		throw std::runtime_error("Failed to create buffers");
	}

	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &lhs_buffer);
	err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &rhs_buffer);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &result_buffer);
	err |= clSetKernelArg(kernel, 3, sizeof(int), &height);
	err |= clSetKernelArg(kernel, 4, sizeof(int), &width);


	if (err != CL_SUCCESS)
	{
		throw std::runtime_error("Failed to set kernel arguments");
	}

	size_t global_work_size[2] = {static_cast<size_t>(height), static_cast<size_t>(width)};
	err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		throw std::runtime_error("Failed to execute kernel");
	}

	// Read results
	err = clEnqueueReadBuffer(queue, result_buffer, CL_TRUE, 0, height * width * sizeof(float), matrix_result, 0, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		throw std::runtime_error("Failed to read results");
	}

	// Cleanup
	clReleaseMemObject(lhs_buffer);
	clReleaseMemObject(rhs_buffer);
	clReleaseMemObject(result_buffer);
	clReleaseKernel(kernel);
	clReleaseProgram(program);

	return matrix_result;

}

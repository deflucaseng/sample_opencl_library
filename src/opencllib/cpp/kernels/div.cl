__kernel void op_kernel(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int rows,
    const int cols
) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        // Add check for division by zero
        C[idx] = (B[idx] != 0.0f) ? A[idx] / B[idx] : 0.0f;
    }
}
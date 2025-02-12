import sample_opencl_lib
import unittest
import time


class OpenCLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize results dictionary with nested structure 
        cls.results = {
            'small_add': [0, 0],      # [CPU_time, GPU_time]
            'medium_add': [0, 0],
            'large_add': [0, 0],
            'small_subtract': [0, 0],  
            'medium_subtract': [0, 0],
            'large_subtract': [0, 0],
            'small_multiply': [0, 0],
            'medium_multiply': [0, 0], 
            'large_multiply': [0, 0],
            'small_divide': [0, 0],
            'medium_divide': [0, 0],
            'large_divide': [0, 0]
        }
        
        # Test matrices
        cls.small_matrix_one = sample_opencl_lib.Matrix([1.0, 2.0, 3.0, 4.0], 2, 2)
        cls.small_matrix_two = sample_opencl_lib.Matrix([5.0, 6.0, 7.0, 8.0], 2, 2)
        
        cls.medium_matrix_one = sample_opencl_lib.Matrix([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3, 3)
        cls.medium_matrix_two = sample_opencl_lib.Matrix([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0], 3, 3)
        
        cls.large_matrix_one = sample_opencl_lib.Matrix([1.0 for _ in range(1000000)], 1000, 1000)
        cls.large_matrix_two = sample_opencl_lib.Matrix([2.0 for _ in range(1000000)], 1000, 1000)
        
        cls.curr_test = ""
        cls.device = ""

    @classmethod
    def tearDownClass(cls):
        print("\nPerformance Results:")
        for test_name, times in cls.results.items():
            print(f"{test_name}:")
            print(f"  CPU: {times[0]:.6f} seconds")
            print(f"  GPU: {times[1]:.6f} seconds")
            print(f"  Speedup: {times[0]/times[1]:.2f}x")

    def setUp(self):
        self._started_at = time.time()

    def tearDown(self):
        elapsed = time.time() - self._started_at
        self.__class__.results[self.__class__.curr_test][1 if self.__class__.device == "GPU" else 0] = elapsed

    # Addition Tests
    def test_small_matrix_add_cpu(self):
        self.__class__.curr_test = "small_add"
        self.__class__.device = "CPU"
        op = sample_opencl_lib.OperationManager("CPU")
        result = op.element_wise_operation("add", self.small_matrix_one, self.small_matrix_two)
        expected = [6.0, 8.0, 10.0, 12.0]
        self.assertEqual(result, expected)

    def test_small_matrix_add_gpu(self):
        self.__class__.curr_test = "small_add"
        self.__class__.device = "GPU"
        op = sample_opencl_lib.OperationManager("GPU")
        result = op.element_wise_operation("add", self.small_matrix_one, self.small_matrix_two)
        expected = [6.0, 8.0, 10.0, 12.0]
        self.assertEqual(result, expected)

    def test_medium_matrix_add_cpu(self):
        self.__class__.curr_test = "medium_add"
        self.__class__.device = "CPU"
        op = sample_opencl_lib.OperationManager("CPU")
        result = op.element_wise_operation("add", self.medium_matrix_one, self.medium_matrix_two)
        expected = [11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0, 25.0, 27.0]
        self.assertEqual(result, expected)

    def test_medium_matrix_add_gpu(self):
        self.__class__.curr_test = "medium_add"
        self.__class__.device = "GPU"
        op = sample_opencl_lib.OperationManager("GPU")
        result = op.element_wise_operation("add", self.medium_matrix_one, self.medium_matrix_two)
        expected = [11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0, 25.0, 27.0]
        self.assertEqual(result, expected)

    def test_large_matrix_add_cpu(self):
        self.__class__.curr_test = "large_add"
        self.__class__.device = "CPU"
        op = sample_opencl_lib.OperationManager("CPU")
        result = op.element_wise_operation("add", self.large_matrix_one, self.large_matrix_two)
        self.assertEqual(len(result), 1000000)
        self.assertEqual(result[0], 3.0)
        self.assertEqual(result[-1], 3.0)
        self.assertEqual(result[500000], 3.0)

    def test_large_matrix_add_gpu(self):
        self.__class__.curr_test = "large_add"
        self.__class__.device = "GPU"
        op = sample_opencl_lib.OperationManager("GPU")
        result = op.element_wise_operation("add", self.large_matrix_one, self.large_matrix_two)
        self.assertEqual(len(result), 1000000)
        self.assertEqual(result[0], 3.0)
        self.assertEqual(result[-1], 3.0)
        self.assertEqual(result[500000], 3.0)

    # Subtraction Tests (existing)
    def test_small_matrix_subtract_cpu(self):
        self.__class__.curr_test = "small_subtract"
        self.__class__.device = "CPU"
        op = sample_opencl_lib.OperationManager("CPU")
        result = op.element_wise_operation("subtract", self.small_matrix_one, self.small_matrix_two)
        
        # Expected result: [1-5, 2-6, 3-7, 4-8] = [-4, -4, -4, -4]
        expected = [-4.0, -4.0, -4.0, -4.0]
        self.assertEqual(result, expected)

    def test_small_matrix_subtract_gpu(self):
        self.__class__.curr_test = "small_subtract"
        self.__class__.device = "GPU"
        op = sample_opencl_lib.OperationManager("GPU")
        result = op.element_wise_operation("subtract", self.small_matrix_one, self.small_matrix_two)
        expected = [-4.0, -4.0, -4.0, -4.0]
        self.assertEqual(result, expected)

    def test_medium_matrix_subtract_cpu(self):
        self.__class__.curr_test = "medium_subtract"
        self.__class__.device = "CPU"
        op = sample_opencl_lib.OperationManager("CPU")
        result = op.element_wise_operation("subtract", self.medium_matrix_one, self.medium_matrix_two)
        expected = [-9.0, -9.0, -9.0, -9.0, -9.0, -9.0, -9.0, -9.0, -9.0]
        self.assertEqual(result, expected)

    def test_medium_matrix_subtract_gpu(self):
        self.__class__.curr_test = "medium_subtract"
        self.__class__.device = "GPU"
        op = sample_opencl_lib.OperationManager("GPU")
        result = op.element_wise_operation("subtract", self.medium_matrix_one, self.medium_matrix_two)
        expected = [-9.0, -9.0, -9.0, -9.0, -9.0, -9.0, -9.0, -9.0, -9.0]
        self.assertEqual(result, expected)

    def test_large_matrix_subtract_cpu(self):
        self.__class__.curr_test = "large_subtract"
        self.__class__.device = "CPU"
        op = sample_opencl_lib.OperationManager("CPU")
        result = op.element_wise_operation("subtract", self.large_matrix_one, self.large_matrix_two)
        # For large matrices, we'll just check a few values and the length
        self.assertEqual(len(result), 1000000)
        self.assertEqual(result[0], -1.0)
        self.assertEqual(result[-1], -1.0)
        self.assertEqual(result[500000], -1.0)

    def test_large_matrix_subtract_gpu(self):
        self.__class__.curr_test = "large_subtract"
        self.__class__.device = "GPU"
        op = sample_opencl_lib.OperationManager("GPU")
        result = op.element_wise_operation("subtract", self.large_matrix_one, self.large_matrix_two)
        self.assertEqual(len(result), 1000000)
        self.assertEqual(result[0], -1.0)
        self.assertEqual(result[-1], -1.0)
        self.assertEqual(result[500000], -1.0)

    # ... (existing subtraction tests)

    # Multiplication Tests
    def test_small_matrix_multiply_cpu(self):
        self.__class__.curr_test = "small_multiply"
        self.__class__.device = "CPU"
        op = sample_opencl_lib.OperationManager("CPU")
        result = op.element_wise_operation("multiply", self.small_matrix_one, self.small_matrix_two)
        expected = [5.0, 12.0, 21.0, 32.0]
        self.assertEqual(result, expected)

    def test_small_matrix_multiply_gpu(self):
        self.__class__.curr_test = "small_multiply"
        self.__class__.device = "GPU"
        op = sample_opencl_lib.OperationManager("GPU")
        result = op.element_wise_operation("multiply", self.small_matrix_one, self.small_matrix_two)
        expected = [5.0, 12.0, 21.0, 32.0]
        self.assertEqual(result, expected)

    def test_medium_matrix_multiply_cpu(self):
        self.__class__.curr_test = "medium_multiply"
        self.__class__.device = "CPU"
        op = sample_opencl_lib.OperationManager("CPU")
        result = op.element_wise_operation("multiply", self.medium_matrix_one, self.medium_matrix_two)
        expected = [10.0, 22.0, 36.0, 52.0, 70.0, 90.0, 112.0, 136.0, 162.0]
        self.assertEqual(result, expected)

    def test_medium_matrix_multiply_gpu(self):
        self.__class__.curr_test = "medium_multiply"
        self.__class__.device = "GPU"
        op = sample_opencl_lib.OperationManager("GPU")
        result = op.element_wise_operation("multiply", self.medium_matrix_one, self.medium_matrix_two)
        expected = [10.0, 22.0, 36.0, 52.0, 70.0, 90.0, 112.0, 136.0, 162.0]
        self.assertEqual(result, expected)

    def test_large_matrix_multiply_cpu(self):
        self.__class__.curr_test = "large_multiply"
        self.__class__.device = "CPU"
        op = sample_opencl_lib.OperationManager("CPU")
        result = op.element_wise_operation("multiply", self.large_matrix_one, self.large_matrix_two)
        self.assertEqual(len(result), 1000000)
        self.assertEqual(result[0], 2.0)
        self.assertEqual(result[-1], 2.0)
        self.assertEqual(result[500000], 2.0)

    def test_large_matrix_multiply_gpu(self):
        self.__class__.curr_test = "large_multiply"
        self.__class__.device = "GPU"
        op = sample_opencl_lib.OperationManager("GPU")
        result = op.element_wise_operation("multiply", self.large_matrix_one, self.large_matrix_two)
        self.assertEqual(len(result), 1000000)
        self.assertEqual(result[0], 2.0)
        self.assertEqual(result[-1], 2.0)
        self.assertEqual(result[500000], 2.0)

    # Division Tests
    def test_small_matrix_divide_cpu(self):
        self.__class__.curr_test = "small_divide"
        self.__class__.device = "CPU"
        op = sample_opencl_lib.OperationManager("CPU")
        result = op.element_wise_operation("divide", self.small_matrix_one, self.small_matrix_two)
        expected = [0.2, 0.333333, 0.428571, 0.5]  # Approximate values
        for r, e in zip(result, expected):
            self.assertAlmostEqual(r, e, places=5)

    def test_small_matrix_divide_gpu(self):
        self.__class__.curr_test = "small_divide"
        self.__class__.device = "GPU"
        op = sample_opencl_lib.OperationManager("GPU")
        result = op.element_wise_operation("divide", self.small_matrix_one, self.small_matrix_two)
        expected = [0.2, 0.333333, 0.428571, 0.5]  # Approximate values
        for r, e in zip(result, expected):
            self.assertAlmostEqual(r, e, places=5)

    def test_medium_matrix_divide_cpu(self):
        self.__class__.curr_test = "medium_divide"
        self.__class__.device = "CPU"
        op = sample_opencl_lib.OperationManager("CPU")
        result = op.element_wise_operation("divide", self.medium_matrix_one, self.medium_matrix_two)
        expected = [0.1, 0.181818, 0.25, 0.307692, 0.357143, 0.4, 0.4375, 0.470588, 0.5]
        for r, e in zip(result, expected):
            self.assertAlmostEqual(r, e, places=5)

    def test_medium_matrix_divide_gpu(self):
        self.__class__.curr_test = "medium_divide"
        self.__class__.device = "GPU"
        op = sample_opencl_lib.OperationManager("GPU")
        result = op.element_wise_operation("divide", self.medium_matrix_one, self.medium_matrix_two)
        expected = [0.1, 0.181818, 0.25, 0.307692, 0.357143, 0.4, 0.4375, 0.470588, 0.5]
        for r, e in zip(result, expected):
            self.assertAlmostEqual(r, e, places=5)

    def test_large_matrix_divide_cpu(self):
        self.__class__.curr_test = "large_divide"
        self.__class__.device = "CPU"
        op = sample_opencl_lib.OperationManager("CPU")
        result = op.element_wise_operation("divide", self.large_matrix_one, self.large_matrix_two)
        self.assertEqual(len(result), 1000000)
        self.assertAlmostEqual(result[0], 0.5, places=5)
        self.assertAlmostEqual(result[-1], 0.5, places=5)
        self.assertAlmostEqual(result[500000], 0.5, places=5)

    def test_large_matrix_divide_gpu(self):
        self.__class__.curr_test = "large_divide"
        self.__class__.device = "GPU"
        op = sample_opencl_lib.OperationManager("GPU")
        result = op.element_wise_operation("divide", self.large_matrix_one, self.large_matrix_two)
        self.assertEqual(len(result), 1000000)
        self.assertAlmostEqual(result[0], 0.5, places=5)
        self.assertAlmostEqual(result[-1], 0.5, places=5)
        self.assertAlmostEqual(result[500000], 0.5, places=5)



if __name__ == '__main__':
    unittest.main()

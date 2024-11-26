#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define THREADS 1024

namespace
{
  __device__ int binary_search(const int *__restrict__ test_elements, int left, int right, int target)
  {
    while (left <= right)
    {
      int mid = left + ((right - left) / 2);
      int val = test_elements[mid];
      if (val == target)
      {
        return mid; // Return the index if found
      }
      else if (val < target)
      {
        left = mid + 1;
      }
      else
      {
        right = mid - 1;
      }
    }
    return -1; // Return -1 if not found
  }

  __global__ void text_sim_kernel(
      const int *__restrict__ elements,
      const int *__restrict__ test_elements,
      const float *__restrict__ element_weights,
      const float *__restrict__ test_element_weights,
      const int64_t batch_size,
      const int64_t elements_size,
      const int64_t num_test,
      const int64_t test_elements_size,
      const int padding_idx,
      float *__restrict__ output)
  {
      const int64_t total_elements = batch_size * elements_size * num_test;
      const int64_t global_index = blockIdx.x * blockDim.x + threadIdx.x;

      if (global_index >= total_elements) return;

      // Corrected indexing
      const int64_t element_index = (global_index / num_test) % elements_size;
      const int64_t batch_index = global_index / (elements_size * num_test);
      const int64_t test_index = global_index % num_test;

      const int64_t element_offset = batch_index * elements_size + element_index;
      const int64_t test_offset = test_index * test_elements_size;

      const int element = elements[element_offset];
      if (element == padding_idx)
          return;

      const int *test_elements_ptr = test_elements + test_offset;
      int found_index = binary_search(test_elements_ptr, 0, test_elements_size - 1, element);

      if (found_index != -1)
      {
          float element_weight = element_weights[element_offset];
          float test_element_weight = test_element_weights[test_offset + found_index];
          float product = element_weight * test_element_weight;

          // Accumulate contributions only once per batch
          atomicAdd(output + batch_index * num_test + test_index, product);
      }
  }

} // namespace

at::Tensor text_sim(
    at::Tensor elements,
    at::Tensor test_elements,
    at::Tensor element_weights,
    at::Tensor test_element_weights,
    int padding_idx)
{
  CHECK_CUDA(elements);
  CHECK_CUDA(test_elements);
  CHECK_CUDA(element_weights);
  CHECK_CUDA(test_element_weights);

  TORCH_CHECK(elements.is_contiguous(), "elements must be contiguous");
  TORCH_CHECK(test_elements.is_contiguous(), "test_elements must be contiguous");
  TORCH_CHECK(element_weights.is_contiguous(), "element_weights must be contiguous");
  TORCH_CHECK(test_element_weights.is_contiguous(), "test_element_weights must be contiguous");

  const auto device = elements.get_device();
  cudaSetDevice(device);

  const int64_t batch_size = elements.size(0);
  const int64_t elements_size = elements.size(1);
  const int64_t num_test = test_elements.size(0);
  const int64_t test_elements_size = test_elements.size(1);

  // Ensure input tensors have correct types
  TORCH_CHECK(elements.scalar_type() == at::kInt, "elements must be int32");
  TORCH_CHECK(test_elements.scalar_type() == at::kInt, "test_elements must be int32");
  TORCH_CHECK(element_weights.scalar_type() == at::kFloat, "element_weights must be float32");
  TORCH_CHECK(test_element_weights.scalar_type() == at::kFloat, "test_element_weights must be float32");

  // Create a float output tensor with shape [B, N]
  auto output = at::zeros({batch_size, num_test}, elements.options().dtype(at::kFloat));

  const int64_t total_threads = batch_size * num_test * elements_size;
  dim3 block(THREADS);
  dim3 grid((total_threads + THREADS - 1) / THREADS);

  // Launch kernel on the current CUDA stream
  text_sim_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      elements.data_ptr<int>(),
      test_elements.data_ptr<int>(),
      element_weights.data_ptr<float>(),
      test_element_weights.data_ptr<float>(),
      batch_size,
      elements_size,
      num_test,
      test_elements_size,
      padding_idx,
      output.data_ptr<float>());

  // Check for CUDA errors
  AT_CUDA_CHECK(cudaGetLastError());

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("text_sim", &text_sim, "Text Similarity CUDA kernel with correct indexing");
}

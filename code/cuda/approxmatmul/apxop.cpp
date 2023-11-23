#include <torch/extension.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_APXTYPE(x) AT_ASSERTM(x >= 0 && x <= 8 , #x " must be betweet 0 and 8")
#define CHECK_THREADS(x) AT_ASSERTM(x >= 0 && x <= 32 , #x " must be lower than 32")

torch::Tensor bmm_cuda(
    torch::Tensor a, // tensor A, converted to int32 in python code
    torch::Tensor b, // tensor B, converted to int32 in python code
    int apx_type, // approximation type
    int threads // number of threads used by GPUs per block
  );

torch::Tensor mm_cuda(
    torch::Tensor a, // tensor A
    torch::Tensor b, // tensor B
    int apx_type,
    int threads
  );

torch::Tensor bmm(
    torch::Tensor a,
    torch::Tensor b,
    int apx_type,
    int threads
  ) {
  CHECK_INPUT(a);
  CHECK_INPUT(b);
  CHECK_APXTYPE(apx_type);
  CHECK_THREADS(threads);
  return bmm_cuda(a, b, apx_type, threads);
}

torch::Tensor mm(
    torch::Tensor a,
    torch::Tensor b,
    int apx_type,
    int threads
  ) {
  CHECK_INPUT(a);
  CHECK_INPUT(b);
  CHECK_APXTYPE(apx_type);
  CHECK_THREADS(threads);
  return mm_cuda(a, b, apx_type, threads);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bmm", &bmm, "BatchMM");
  m.def("mm", &mm, "MM");
}
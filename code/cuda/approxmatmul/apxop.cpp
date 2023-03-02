#include <torch/extension.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_APXTYPE(x) AT_ASSERTM(x >= 0 && x <= 8 , #x " must be betweet 0 and 8")

torch::Tensor bmm_cuda(
    torch::Tensor a, // tensor A
    torch::Tensor b, // tensor B
    int apx_type
  );

torch::Tensor mm_cuda(
    torch::Tensor a, // tensor A
    torch::Tensor b, // tensor B
    int apx_type
  );

torch::Tensor bmm(
    torch::Tensor a,
    torch::Tensor b,
    int apx_type
  ) {
  CHECK_INPUT(a);
  CHECK_INPUT(b);
  CHECK_APXTYPE(apx_type);
  return bmm_cuda(a, b, apx_type);
}

torch::Tensor mm(
    torch::Tensor a,
    torch::Tensor b,
    int apx_type
  ) {
  CHECK_INPUT(a);
  CHECK_INPUT(b);
  CHECK_APXTYPE(apx_type);
  return mm_cuda(a, b, apx_type);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bmm", &bmm, "BatchMM");
  m.def("mm", &mm, "MM");
}

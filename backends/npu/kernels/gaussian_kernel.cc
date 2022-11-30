// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"
#include "kernels/funcs/op_command.h"

namespace custom_kernel {

template <typename T, typename Context>
void GaussianKernel(const Context& ctx,
                    const phi::IntArray& shape,
                    float mean,
                    float std,
                    int seed,
                    phi::DataType dtype,
                    phi::DenseTensor* out) {
  ctx.template Alloc<T>(out);

  phi::DenseTensor cpu_tensor;
  cpu_tensor.Resize(out->dims());
  T* cpu_data = ctx.template HostAlloc<T>(&cpu_tensor);
  std::normal_distribution<T> dist(mean, std);

  int64_t size = out->numel();

  auto gen_ptr = ctx.GetGenerator();
  gen_ptr->SetCurrentSeed(static_cast<int64_t>(seed));
  auto engine = gen_ptr->GetCPUEngine();

  for (int64_t i = 0; i < size; ++i) {
    cpu_data[i] = dist(*engine);
  }
  ACL_RUN({ TensorCopy(ctx, cpu_tensor, false, out); });
  GRAPH_RUN({
    experimental::OpCommand("Const")
        .Output(
            *out,
            experimental::TensorDescMaker("y").FromTensor(*out).SetDataLayout(
                phi::DataLayout::ANY))
        .Attr("value", cpu_tensor)
        .Run(ctx);
  });
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(
    gaussian, npu, ALL_LAYOUT, custom_kernel::GaussianKernel, float) {}

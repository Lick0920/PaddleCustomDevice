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

namespace custom_kernel {

template <typename T, typename Context>
void UniqueKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  bool return_index,
                  bool return_inverse,
                  bool return_counts,
                  const std::vector<int>& axis,
                  DataType dtype,
                  phi::DenseTensor* out,
                  phi::DenseTensor* indices,
                  phi::DenseTensor* index,
                  phi::DenseTensor* counts) {
  bool is_sorted = true;
  UniqueRawKernel<T, Context>(dev_ctx,
                              x,
                              return_index,
                              return_inverse,
                              return_counts,
                              axis,
                              dtype,
                              is_sorted,
                              out,
                              indices,
                              index,
                              counts);
}

template <typename T, typename Context>
void UniqueRawKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     bool return_index,
                     bool return_inverse,
                     bool return_counts,
                     const std::vector<int>& axis,
                     DataType dtype,
                     bool is_sorted,
                     phi::DenseTensor* out,
                     phi::DenseTensor* indices,
                     phi::DenseTensor* index,
                     phi::DenseTensor* counts){
  if (dtype == phi::DataType::INT32) {
    PADDLE_ENFORCE_LE(
        x.numel(),
        INT_MAX,
        phi::errors::InvalidArgument(
            "The number of elements in Input(X) should be less than or "
            "equal to INT_MAX, but received num is %d. Please set `dtype` to "
            "int64.",
            x.numel()));
  }
  // if (!is_sorted) {
  //   phi::VisitDataType(
  //       dtype,
  //       phi::funcs::UniqueOpFunctor<Context, T>(dev_ctx, out, index, &x));
  //   return;
  // }

  if (axis.empty()) {
    // 怎么知道独有元素构成的tensor的维度是多少？
    // phi::DDim output_dims = x.dims();
    // out->Resize(output_dims);
    dev_ctx.template Alloc<T>(out); // ???
    auto npu_stream = dev_ctx.stream();
    NpuOpRunner npu_op_runner_unique;
    npu_op_runner_unique.SetType("Unique")
        .AddInput(x)
        .AddOutput(*out)
        .Run(npu_stream);
    
  } else {
    auto npu_stream = dev_ctx.stream();
    printf("axis: %d\n", axis[0]);
}
                    
} // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(unique,
                   npu,
                   ALL_LAYOUT,
                   custom_kernel::UniqueKernel,
                   float,
                   double,
                   int32_t,
                   int64_t) {
  kernel->OutputAt(1).SetDataType(phi::DataType::UNDEFINED);
  kernel->OutputAt(2).SetDataType(phi::DataType::UNDEFINED);
  kernel->OutputAt(3).SetDataType(phi::DataType::UNDEFINED);
}

PD_REGISTER_PLUGIN_KERNEL(unique_raw,
                   npu,
                   ALL_LAYOUT,
                   custom_kernel::UniqueRawKernel,
                   float,
                   double,
                   int32_t,
                   int64_t) {
  kernel->OutputAt(1).SetDataType(phi::DataType::UNDEFINED);
  kernel->OutputAt(2).SetDataType(phi::DataType::UNDEFINED);
  kernel->OutputAt(3).SetDataType(phi::DataType::UNDEFINED);
}

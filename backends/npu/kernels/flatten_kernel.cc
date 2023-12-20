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
void FlattenInferKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        int start_axis UNUSED,
                        int stop_axis UNUSED,
                        phi::DenseTensor* out) {
  dev_ctx.Alloc(out, x.dtype());
  auto out_dims = out->dims();
  phi::Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);
  out->Resize(out_dims);
}

template <typename T, typename Context>
void FlattenKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   int start_axis,
                   int stop_axis,
                   phi::DenseTensor* out,
                   phi::DenseTensor* xshape UNUSED){
  FlattenInferKernel<T, Context>(dev_ctx, x, start_axis, stop_axis, out);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(flatten_infer,
                   npu,
                   ALL_LAYOUT,
                   phi::FlattenInferKernel,
                   float,
                   phi::dtype::bfloat16,
                   double,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {}

PD_REGISTER_PLUGIN_KERNEL(flatten,
                   npu,
                   ALL_LAYOUT,
                   phi::FlattenKernel,
                   float,
                   phi::dtype::bfloat16,
                   double,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {}

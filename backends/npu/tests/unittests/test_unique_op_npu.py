#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import numpy as np
import unittest

from tests.op_test import OpTest
import paddle

paddle.enable_static()


class TestUniqueOp(OpTest):
    def setUp(self):
        self.set_npu()
        self.set_dtype()
        self.set_data()
        self.set_attrs()
        self.op_type = "unique"
        self.inputs = {"X": self.input_data}
        self.outputs = {
            "Out": self.out_data,
            "Index": self.index_data,
            "Inverse" : self.inverse_data,
            "Counts": self.count_data
        }

    def set_dtype(self):
        self.dtype = np.int32

    def set_npu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("npu", 0)

    def set_data(self):
        arr = np.array([1,1,2,3,1,4,5,6,6])
        arr_o = np.array([1,2,3,4,5,6])
        dtype = np.int32  # 默认数据类型为int32
        self.input_data = arr.astype(dtype)
        self.out_data = arr_o.astype(dtype)
        self.index_data = np.array([0,2,3,5,6,7]).astype(
            np.int32)  # 默认数据类型为int32
        self.inverse_data = np.array([0,0,1,2,0,3,4,5,5]).astype(
            np.int32)  # 默认数据类型为int32
        self.count_data = np.array([3,1,1,1,1,2]).astype(
            np.int32)  # 默认数据类型为int32

    def test_check_output(self):
        self.check_output_with_place(self.place)
        

    def set_attrs(self):
        pass

    # @unittest.skip("skip check_grad because unstable.")
    # def test_check_grad(self):
    #     self.check_grad_with_place(self.place, ["X"], "Out")


# # Correct: There is mins axis.
# class TestUnsqueeze2Op1(TestUnsqueeze2Op):
#     def init_test_case(self):
#         self.ori_shape = (20, 5)
#         self.axes = (0, -2)
#         self.new_shape = (1, 20, 1, 5)



if __name__ == "__main__":
    unittest.main()

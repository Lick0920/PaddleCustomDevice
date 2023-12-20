# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
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



class TestFlatten(OpTest):
    def setUp(self):
        self.op_type = "flatten"
        self.python_api = paddle.flatten
        self.public_python_api = paddle.flatten
        self.python_out_sig = ["Out"]
        self.start_axis = 0
        self.stop_axis = -1
        self.set_npu()
        self.init_test_case()
        self.init_dtype()
        self.init_input_data()
        self.init_attrs()
        self.outputs = {
            "Out": self.inputs["X"].reshape(self.new_shape),
            "XShape": np.random.random(self.in_shape).astype("float32"),
        }

    def set_npu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("npu", 0)

    def test_check_output(self):
        self.check_output_with_place(self.place, no_check_set=["XShape"])

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Out")

    def init_test_case(self):
        self.in_shape = (3, 2, 5, 4)
        self.start_axis = 0
        self.stop_axis = -1
        self.new_shape = 120

    def init_attrs(self):
        self.attrs = {
            "start_axis": self.start_axis,
            "stop_axis": self.stop_axis,
        }

    def init_dtype(self):
        self.dtype = "float32"

    def init_input_data(self):
        x = np.random.random(self.in_shape).astype(self.dtype)
        self.inputs = {"X": x}


class TestFlatten2(TestFlatten):
    def init_dtype(self):
        self.dtype = "int32"


if __name__ == "__main__":
    unittest.main()

#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import numpy as np
from op_test import OpTest

import paddle
from paddle.base import core
from paddle.pir_utils import test_with_pir_api

paddle.enable_static()


class TestTruncOp(OpTest):
    def setUp(self):
        self.op_type = "trunc"
        self.set_npu()
        self.python_api = paddle.trunc
        self.init_dtype_type()
        np.random.seed(2021)
        self.inputs = {'X': np.random.random((20, 20)).astype(self.dtype)}
        self.outputs = {'Out': (np.trunc(self.inputs['X']))}

    def init_dtype_type(self):
        self.dtype = np.float32
    
    def set_npu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("npu", 0)


    def test_check_output(self):
        self.__class__.no_need_check_grad = True
        self.check_output_with_place(self.place)



class TestIntTruncOp(TestTruncOp):
    def init_dtype_type(self):
        self.dtype = np.int32


class TestTruncAPI(unittest.TestCase):
    def setUp(self):
        self.shape = [20, 20]
        self.x = np.random.random((20, 20)).astype(np.float32)
        self.place = paddle.CPUPlace()

    @test_with_pir_api
    def test_api_static(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', self.shape)
            out = paddle.trunc(x)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x}, fetch_list=[out])
        out_ref = np.trunc(self.x)
        for out in res:
            np.testing.assert_allclose(out, out_ref, rtol=1e-08)

    def test_api_dygraph(self):
        paddle.disable_static(self.place)
        x_tensor = paddle.to_tensor(self.x)
        out = paddle.trunc(x_tensor)
        out_ref = np.trunc(self.x)
        np.testing.assert_allclose(out.numpy(), out_ref, rtol=1e-08)
        paddle.enable_static()

    def test_errors(self):
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', [20, 20], 'bool')
            self.assertRaises(TypeError, paddle.trunc, x)



class TestTruncFP16OP(TestTruncOp):
    def init_dtype_type(self):
        self.dtype = np.float16



if __name__ == "__main__":
    unittest.main()

# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc

import warp as wp

wp.config.enable_backward = False

from asv_runner.benchmarks.mark import skip_benchmark_if

from newton.examples.example_mujoco import Example


class KpiInitializeModel:
    params = (["humanoid", "g1", "h1", "cartpole", "ant", "quadruped"], [4096, 8192])
    param_names = ["robot", "num_envs"]

    rounds = 1
    number = 1
    repeat = 3
    min_run_count = 1
    timeout = 3600

    def setup(self, robot, num_envs):
        wp.init()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_initialize_model(self, robot, num_envs):
        builder = Example.create_model_builder(robot, num_envs, randomize=True, seed=123)

        # finalize model
        _model = builder.finalize()
        wp.synchronize_device()


class FastInitializeModel:
    params = (["humanoid", "g1", "h1", "cartpole", "ant", "quadruped"], [128, 256])
    param_names = ["robot", "num_envs"]

    rounds = 1
    number = 1
    repeat = 3
    min_run_count = 1

    def setup_cache(self):
        # Load a small model to cache the kernels
        builder = Example.create_model_builder("cartpole", 1, randomize=False, seed=123)
        model = builder.finalize(device="cpu")
        del model

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_initialize_model(self, robot, num_envs):
        builder = Example.create_model_builder(robot, num_envs, randomize=True, seed=123)

        # finalize model
        _model = builder.finalize()
        wp.synchronize_device()

    def peakmem_initialize_model_cpu(self, robot, num_envs):
        gc.collect()

        with wp.ScopedDevice("cpu"):
            builder = Example.create_model_builder(robot, num_envs, randomize=True, seed=123)

            # finalize model
            model = builder.finalize()

        del model

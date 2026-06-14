#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""
Data processors for MGP (Markov Generator Policy).

Inherits from DiffusionPolicy processors - MGP uses same preprocessing.
"""

from lerobot.policies.diffusion.processor_diffusion import make_diffusion_pre_post_processors


def make_mgp_pre_post_processors(config, dataset_stats=None):
    """
    Create preprocessing and postprocessing functions for MGP.

    MGP uses the same data processing as DiffusionPolicy.
    This is a convenience wrapper for consistency with other policies.

    Args:
        config: MGPConfig instance
        dataset_stats: Dataset statistics for normalization

    Returns:
        (preprocess_fn, postprocess_fn): Preprocessing and postprocessing functions
    """
    return make_diffusion_pre_post_processors(config, dataset_stats)

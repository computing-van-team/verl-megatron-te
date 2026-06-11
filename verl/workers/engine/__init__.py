# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
try:
    # Apply Ascend NPU monkey patches before any megatron import below (.fsdp imports
    # megatron transitively). Broad except: torch_npu raises non-ImportError when CANN
    # env is missing, which must not break non-NPU backends.
    import megatron_adaptor  # noqa: F401
except Exception:
    pass

from .base import BaseEngine, EngineRegistry
from .fsdp import FSDPEngine, FSDPEngineWithLMHead

__all__ = [
    "BaseEngine",
    "EngineRegistry",
    "FSDPEngine",
    "FSDPEngineWithLMHead",
]

try:
    from .torchtitan import TorchTitanEngine, TorchTitanEngineWithLMHead

    __all__ += ["TorchTitanEngine", "TorchTitanEngineWithLMHead"]
except ImportError:
    TorchTitanEngine = None
    TorchTitanEngineWithLMHead = None

try:
    from .veomni import VeOmniEngine, VeOmniEngineWithLMHead

    __all__ += ["VeOmniEngine", "VeOmniEngineWithLMHead"]
except ImportError:
    VeOmniEngine = None
    VeOmniEngineWithLMHead = None

try:
    from .automodel import AutomodelEngine, AutomodelEngineWithLMHead

    __all__ += ["AutomodelEngine", "AutomodelEngineWithLMHead"]
except ImportError:
    AutomodelEngine = None
    AutomodelEngineWithLMHead = None

# MegatronAdaptor (Ascend NPU) must be imported before Megatron so its monkey patches
# are in effect when megatron modules are first imported
try:
    from .megatron_adaptor import MegatronAdaptorEngineWithLMHead, MegatronAdaptorEngineWithValueHead

    __all__ += ["MegatronAdaptorEngineWithLMHead", "MegatronAdaptorEngineWithValueHead"]
except ImportError:
    MegatronAdaptorEngineWithLMHead = None
    MegatronAdaptorEngineWithValueHead = None

# Mindspeed must be imported before Megatron to ensure the related monkey patches take effect as expected
try:
    from .mindspeed import MindspeedEngineWithLMHead, MindspeedEngineWithValueHead, MindSpeedMegatronEngineWithLMHead

    __all__ += ["MindspeedEngineWithLMHead", "MindspeedEngineWithValueHead", "MindSpeedMegatronEngineWithLMHead"]
except ImportError:
    MindspeedEngineWithLMHead = None
    MindspeedEngineWithValueHead = None
    MindSpeedMegatronEngineWithLMHead = None

try:
    from .megatron import MegatronEngine, MegatronEngineWithLMHead, MegatronEngineWithValueHead

    __all__ += ["MegatronEngine", "MegatronEngineWithLMHead", "MegatronEngineWithValueHead"]
except ImportError:
    MegatronEngine = None
    MegatronEngineWithLMHead = None

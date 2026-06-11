# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

import logging
import os

try:
    # Importing megatron_adaptor applies the Ascend NPU monkey patches to megatron.core.
    # It must run before any megatron module is imported so that from-imports elsewhere
    # bind the patched attributes.
    import megatron_adaptor  # noqa: F401

    HAVE_MEGATRON_ADAPTOR = True
except Exception:
    # torch_npu raises non-ImportError (e.g. UnicodeDecodeError) when CANN env is missing
    HAVE_MEGATRON_ADAPTOR = False

from verl.trainer.config import CheckpointConfig
from verl.workers.config import (
    HFModelConfig,
    McoreEngineConfig,
    McoreOptimizerConfig,
)

from ..base import EngineRegistry
from ..megatron import MegatronEngineWithLMHead, MegatronEngineWithValueHead

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_ADAPTOR_REQUIRED_MSG = (
    "backend='megatron_adaptor' requires the megatron_adaptor package "
    "(https://gitcode.com/Ascend/MegatronAdaptor) to be installed"
)


@EngineRegistry.register(model_type="language_model", backend="megatron_adaptor", device="npu")
class MegatronAdaptorEngineWithLMHead(MegatronEngineWithLMHead):
    """Megatron engine on Ascend NPU via the lightweight MegatronAdaptor patch layer.

    Unlike the MindSpeed backend, all patching happens once at import time of
    megatron_adaptor; no per-engine repatch call is needed.
    """

    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: McoreEngineConfig,
        optimizer_config: McoreOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        assert HAVE_MEGATRON_ADAPTOR, _ADAPTOR_REQUIRED_MSG
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)


@EngineRegistry.register(model_type="value_model", backend="megatron_adaptor", device="npu")
class MegatronAdaptorEngineWithValueHead(MegatronEngineWithValueHead):
    """Value-model variant of the MegatronAdaptor NPU engine."""

    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: McoreEngineConfig,
        optimizer_config: McoreOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        assert HAVE_MEGATRON_ADAPTOR, _ADAPTOR_REQUIRED_MSG
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)

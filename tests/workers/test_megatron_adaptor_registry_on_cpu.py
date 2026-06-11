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


def test_registry_resolves_adaptor_engine_on_npu(monkeypatch):
    monkeypatch.setenv("VERL_ENGINE_DEVICE", "npu")
    from verl.workers.engine import EngineRegistry

    lm_cls = EngineRegistry.get_engine_cls("language_model", "megatron_adaptor")
    assert lm_cls.__name__ == "MegatronAdaptorEngineWithLMHead"
    vm_cls = EngineRegistry.get_engine_cls("value_model", "megatron_adaptor")
    assert vm_cls.__name__ == "MegatronAdaptorEngineWithValueHead"


def test_adaptor_does_not_shadow_plain_megatron_backend(monkeypatch):
    monkeypatch.setenv("VERL_ENGINE_DEVICE", "npu")
    from verl.workers.engine import EngineRegistry

    cls = EngineRegistry.get_engine_cls("language_model", "megatron")
    assert cls.__name__ != "MegatronAdaptorEngineWithLMHead"


def test_mcore_config_accepts_adaptor_strategy():
    from verl.workers.config import McoreEngineConfig

    cfg = McoreEngineConfig(strategy="megatron_adaptor", tensor_model_parallel_size=2)
    assert cfg.strategy == "megatron_adaptor"


def test_mcore_config_rejects_unknown_strategy():
    import pytest

    from verl.workers.config import McoreEngineConfig

    with pytest.raises(AssertionError):
        McoreEngineConfig(strategy="not_a_backend")

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for vllm/sampling_params.py"""

from unittest.mock import patch

import pytest

from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import (
    BeamSearchParams,
    RepetitionDetectionParams,
    RequestOutputKind,
    SamplingParams,
    SamplingType,
    StructuredOutputsParams,
)

pytestmark = pytest.mark.cpu_test


# =============================================================================
# StructuredOutputsParams tests
# =============================================================================


class TestStructuredOutputsParams:
    def test_json_constraint(self):
        params = StructuredOutputsParams(json='{"type": "object"}')
        assert params.json == '{"type": "object"}'
        assert not params.all_constraints_none()

    def test_regex_constraint(self):
        params = StructuredOutputsParams(regex=r"\d+")
        assert params.regex == r"\d+"
        assert not params.all_constraints_none()

    def test_choice_constraint(self):
        params = StructuredOutputsParams(choice=["yes", "no"])
        assert params.choice == ["yes", "no"]
        assert not params.all_constraints_none()

    def test_grammar_constraint(self):
        params = StructuredOutputsParams(grammar="root ::= 'hello'")
        assert params.grammar == "root ::= 'hello'"
        assert not params.all_constraints_none()

    def test_json_object_constraint(self):
        params = StructuredOutputsParams(json_object=True)
        assert params.json_object is True
        assert not params.all_constraints_none()

    def test_structural_tag_constraint(self):
        params = StructuredOutputsParams(structural_tag="<tag>")
        assert params.structural_tag == "<tag>"
        assert not params.all_constraints_none()
        assert params.all_non_structural_tag_constraints_none()

    def test_multiple_constraints_raises(self):
        with pytest.raises(ValueError, match="multiple are specified"):
            StructuredOutputsParams(json='{"type": "object"}', regex=r"\d+")

    def test_no_constraint_raises(self):
        with pytest.raises(ValueError, match="none are specified"):
            StructuredOutputsParams()

    def test_all_constraints_none_with_json(self):
        params = StructuredOutputsParams(json='{"type": "object"}')
        assert not params.all_constraints_none()

    def test_all_non_structural_tag_constraints_none(self):
        params = StructuredOutputsParams(structural_tag="<tag>")
        assert params.all_non_structural_tag_constraints_none()

    def test_all_non_structural_tag_constraints_not_none(self):
        params = StructuredOutputsParams(regex=r"\d+")
        assert not params.all_non_structural_tag_constraints_none()

    def test_options_fields(self):
        params = StructuredOutputsParams(
            json='{"type": "object"}',
            disable_any_whitespace=True,
            disable_additional_properties=True,
            whitespace_pattern=r"\s*",
        )
        assert params.disable_any_whitespace is True
        assert params.disable_additional_properties is True
        assert params.whitespace_pattern == r"\s*"


# =============================================================================
# RepetitionDetectionParams tests
# =============================================================================


class TestRepetitionDetectionParams:
    def test_valid_params(self):
        params = RepetitionDetectionParams(
            max_pattern_size=5, min_pattern_size=2, min_count=3
        )
        assert params.max_pattern_size == 5
        assert params.min_pattern_size == 2
        assert params.min_count == 3

    def test_disabled_by_default(self):
        params = RepetitionDetectionParams()
        assert params.max_pattern_size == 0
        assert params.min_pattern_size == 0
        assert params.min_count == 0

    def test_negative_max_pattern_size_raises(self):
        with pytest.raises(ValueError, match="max_pattern_size"):
            RepetitionDetectionParams(max_pattern_size=-1, min_count=2)

    def test_negative_min_pattern_size_raises(self):
        with pytest.raises(ValueError, match="max_pattern_size"):
            RepetitionDetectionParams(
                max_pattern_size=5, min_pattern_size=-1, min_count=2
            )

    def test_min_greater_than_max_raises(self):
        with pytest.raises(ValueError, match="min_pattern_size <= max_pattern_size"):
            RepetitionDetectionParams(
                max_pattern_size=2, min_pattern_size=5, min_count=2
            )

    def test_min_count_less_than_2_raises(self):
        with pytest.raises(ValueError, match="min_count must be >= 2"):
            RepetitionDetectionParams(max_pattern_size=5, min_count=1)


# =============================================================================
# SamplingParams construction and validation tests
# =============================================================================


class TestSamplingParamsConstruction:
    def test_default_values(self):
        params = SamplingParams()
        assert params.n == 1
        assert params.temperature == 1.0
        assert params.top_p == 1.0
        assert params.top_k == 0
        assert params.min_p == 0.0
        assert params.max_tokens == 16
        assert params.min_tokens == 0
        assert params.seed is None
        assert params.stop == []
        assert params.stop_token_ids == []
        assert params.bad_words == []
        assert params.presence_penalty == 0.0
        assert params.frequency_penalty == 0.0
        assert params.repetition_penalty == 1.0

    def test_stop_string_to_list(self):
        params = SamplingParams(stop="hello")
        assert params.stop == ["hello"]

    def test_stop_list_preserved(self):
        params = SamplingParams(stop=["hello", "world"])
        assert params.stop == ["hello", "world"]

    def test_stop_none_becomes_empty_list(self):
        params = SamplingParams(stop=None)
        assert params.stop == []

    def test_stop_token_ids_none_becomes_empty_list(self):
        params = SamplingParams(stop_token_ids=None)
        assert params.stop_token_ids == []

    def test_bad_words_none_becomes_empty_list(self):
        params = SamplingParams(bad_words=None)
        assert params.bad_words == []

    def test_seed_negative_one_becomes_none(self):
        params = SamplingParams(seed=-1)
        assert params.seed is None

    def test_logprobs_true_becomes_one(self):
        params = SamplingParams(logprobs=True)
        assert params.logprobs == 1

    def test_prompt_logprobs_true_becomes_one(self):
        params = SamplingParams(prompt_logprobs=True)
        assert params.prompt_logprobs == 1

    def test_output_text_buffer_length_with_stop(self):
        params = SamplingParams(stop=["hello", "ab"])
        # max len is 5 ("hello") so buffer_length = 5 - 1 = 4
        assert params.output_text_buffer_length == 4

    def test_output_text_buffer_length_with_include_stop(self):
        params = SamplingParams(stop=["hello"], include_stop_str_in_output=True)
        assert params.output_text_buffer_length == 0

    def test_greedy_sampling_resets_top_p_top_k_min_p(self):
        params = SamplingParams(temperature=0.0, top_p=0.9, top_k=5, min_p=0.1)
        assert params.top_p == 1.0
        assert params.top_k == 0
        assert params.min_p == 0.0


class TestSamplingParamsValidation:
    def test_n_less_than_1_raises(self):
        with pytest.raises(ValueError, match="n must be at least 1"):
            SamplingParams(n=0)

    def test_n_not_int_raises(self):
        with pytest.raises(ValueError, match="n must be an int"):
            SamplingParams(n=1.5)

    def test_presence_penalty_out_of_range_raises(self):
        with pytest.raises(ValueError, match="presence_penalty must be in"):
            SamplingParams(presence_penalty=3.0)

    def test_frequency_penalty_out_of_range_raises(self):
        with pytest.raises(ValueError, match="frequency_penalty must be in"):
            SamplingParams(frequency_penalty=-3.0)

    def test_repetition_penalty_zero_raises(self):
        with pytest.raises(ValueError, match="repetition_penalty must be greater"):
            SamplingParams(repetition_penalty=0.0)

    def test_negative_temperature_raises(self):
        with pytest.raises(
            VLLMValidationError, match="temperature must be non-negative"
        ):
            SamplingParams(temperature=-1.0)

    def test_top_p_zero_raises(self):
        with pytest.raises(VLLMValidationError, match="top_p must be in"):
            SamplingParams(top_p=0.0)

    def test_top_p_greater_than_1_raises(self):
        with pytest.raises(VLLMValidationError, match="top_p must be in"):
            SamplingParams(top_p=1.5)

    def test_top_k_less_than_negative_1_raises(self):
        with pytest.raises(ValueError, match="top_k must be 0"):
            SamplingParams(top_k=-2)

    def test_top_k_not_int_raises(self):
        with pytest.raises(TypeError, match="top_k must be an integer"):
            SamplingParams(top_k=1.5)

    def test_min_p_negative_raises(self):
        with pytest.raises(ValueError, match="min_p must be in"):
            SamplingParams(min_p=-0.1)

    def test_min_p_greater_than_1_raises(self):
        with pytest.raises(ValueError, match="min_p must be in"):
            SamplingParams(min_p=1.5)

    def test_max_tokens_zero_raises(self):
        with pytest.raises(VLLMValidationError, match="max_tokens must be at least 1"):
            SamplingParams(max_tokens=0)

    def test_min_tokens_negative_raises(self):
        with pytest.raises(
            ValueError, match="min_tokens must be greater than or equal"
        ):
            SamplingParams(min_tokens=-1)

    def test_min_tokens_greater_than_max_tokens_raises(self):
        with pytest.raises(ValueError, match="min_tokens must be less than or equal"):
            SamplingParams(min_tokens=20, max_tokens=10)

    def test_logprobs_invalid_raises(self):
        with pytest.raises(VLLMValidationError, match="logprobs must be non-negative"):
            SamplingParams(logprobs=-2)

    def test_prompt_logprobs_invalid_raises(self):
        with pytest.raises(
            VLLMValidationError, match="prompt_logprobs must be non-negative"
        ):
            SamplingParams(prompt_logprobs=-2)

    def test_stop_token_ids_non_int_raises(self):
        with pytest.raises(
            ValueError, match="stop_token_ids must contain only integers"
        ):
            SamplingParams(stop_token_ids=[1, "a"])

    def test_empty_stop_string_raises(self):
        with pytest.raises(ValueError, match="stop cannot contain an empty string"):
            SamplingParams(stop=["hello", ""])

    def test_stop_with_detokenize_false_raises(self):
        with pytest.raises(
            ValueError, match="stop strings are only supported when detokenize"
        ):
            SamplingParams(stop=["hello"], detokenize=False)

    def test_greedy_with_n_greater_than_1_raises(self):
        with pytest.raises(ValueError, match="n must be 1 when using greedy"):
            SamplingParams(temperature=0.0, n=2)


class TestSamplingParamsSamplingType:
    def test_greedy_sampling_type(self):
        params = SamplingParams(temperature=0.0)
        assert params.sampling_type == SamplingType.GREEDY

    def test_random_sampling_type(self):
        params = SamplingParams(temperature=0.8)
        assert params.sampling_type == SamplingType.RANDOM

    def test_random_seed_sampling_type(self):
        params = SamplingParams(temperature=0.8, seed=42)
        assert params.sampling_type == SamplingType.RANDOM_SEED


class TestSamplingParamsClone:
    def test_deep_clone(self):
        params = SamplingParams(stop=["hello"], temperature=0.5)
        cloned = params.clone()
        assert cloned is not params
        assert cloned.stop == params.stop
        assert cloned.temperature == params.temperature

    def test_shallow_clone_with_skip_clone(self):
        params = SamplingParams(temperature=0.5, skip_clone=True)
        cloned = params.clone()
        # shallow copy returns a different object but shares internal refs
        assert cloned is not params
        assert cloned.temperature == params.temperature


class TestSamplingParamsFromOptional:
    def test_from_optional_with_none_values(self):
        params = SamplingParams.from_optional(
            n=None,
            presence_penalty=None,
            frequency_penalty=None,
            repetition_penalty=None,
            temperature=None,
            top_p=None,
        )
        assert params.n == 1
        assert params.presence_penalty == 0.0
        assert params.frequency_penalty == 0.0
        assert params.repetition_penalty == 1.0
        assert params.temperature == 1.0
        assert params.top_p == 1.0

    def test_from_optional_logit_bias_clamping(self):
        params = SamplingParams.from_optional(
            logit_bias={"0": 200.0, "1": -200.0, "2": 50.0}
        )
        assert params.logit_bias[0] == 100.0
        assert params.logit_bias[1] == -100.0
        assert params.logit_bias[2] == 50.0

    def test_from_optional_logit_bias_str_keys_to_int(self):
        params = SamplingParams.from_optional(logit_bias={"10": 5.0, "20": -3.0})
        assert 10 in params.logit_bias
        assert 20 in params.logit_bias


class TestSamplingParamsUpdateFromGenerationConfig:
    def test_update_eos_token_id(self):
        params = SamplingParams()
        params.update_from_generation_config({}, eos_token_id=50256)
        assert params.eos_token_id == 50256
        assert 50256 in params.all_stop_token_ids

    def test_update_eos_token_id_with_ignore_eos(self):
        params = SamplingParams(ignore_eos=True)
        params.update_from_generation_config({}, eos_token_id=50256)
        assert params.eos_token_id is None
        # Still added to all_stop_token_ids for min_tokens processing
        assert 50256 in params.all_stop_token_ids

    def test_update_with_multiple_eos_ids(self):
        params = SamplingParams()
        gen_config = {"eos_token_id": [50256, 50257]}
        params.update_from_generation_config(gen_config, eos_token_id=50256)
        # 50256 is the primary, 50257 is added to stop_token_ids
        assert 50257 in params.all_stop_token_ids

    def test_update_with_single_eos_id_in_config(self):
        params = SamplingParams()
        gen_config = {"eos_token_id": 99}
        params.update_from_generation_config(gen_config, eos_token_id=50256)
        assert 99 in params.all_stop_token_ids


class TestSamplingParamsRepr:
    def test_repr_contains_key_fields(self):
        params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=100)
        repr_str = repr(params)
        assert "temperature=0.7" in repr_str
        assert "top_p=0.9" in repr_str
        assert "max_tokens=100" in repr_str


class TestSamplingParamsTemperatureWarning:
    def test_small_temperature_clamped(self):
        # temperature between 0 and _MAX_TEMP (1e-2) is clamped
        params = SamplingParams(temperature=1e-4)
        assert params.temperature >= 1e-2


# =============================================================================
# BeamSearchParams tests
# =============================================================================


class TestBeamSearchParams:
    def test_construction(self):
        params = BeamSearchParams(beam_width=4, max_tokens=100)
        assert params.beam_width == 4
        assert params.max_tokens == 100
        assert params.ignore_eos is False
        assert params.temperature == 0.0
        assert params.length_penalty == 1.0

    def test_custom_values(self):
        params = BeamSearchParams(
            beam_width=8,
            max_tokens=200,
            ignore_eos=True,
            temperature=0.5,
            length_penalty=0.8,
            include_stop_str_in_output=True,
        )
        assert params.beam_width == 8
        assert params.max_tokens == 200
        assert params.ignore_eos is True
        assert params.temperature == 0.5
        assert params.length_penalty == 0.8
        assert params.include_stop_str_in_output is True


# =============================================================================
# RequestOutputKind tests
# =============================================================================


class TestRequestOutputKind:
    def test_enum_values(self):
        assert RequestOutputKind.CUMULATIVE.value == 0
        assert RequestOutputKind.DELTA.value == 1
        assert RequestOutputKind.FINAL_ONLY.value == 2


# =============================================================================
# SamplingParams with VLLM_MAX_N_SEQUENCES env var
# =============================================================================


class TestSamplingParamsMaxN:
    def test_n_exceeds_max_raises(self):
        with (
            patch("vllm.envs.VLLM_MAX_N_SEQUENCES", 10),
            pytest.raises(ValueError, match="n must be at most 10"),
        ):
            SamplingParams(n=11)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Extended unit tests for vllm/outputs.py"""

import pytest
import torch

from vllm.outputs import (
    ClassificationOutput,
    ClassificationRequestOutput,
    CompletionOutput,
    EmbeddingOutput,
    EmbeddingRequestOutput,
    PoolingOutput,
    PoolingRequestOutput,
    RequestOutput,
    ScoringOutput,
    ScoringRequestOutput,
)

pytestmark = pytest.mark.cpu_test


# =============================================================================
# CompletionOutput tests
# =============================================================================


class TestCompletionOutput:
    def test_construction(self):
        output = CompletionOutput(
            index=0,
            text="Hello world",
            token_ids=[1, 2, 3],
            cumulative_logprob=-1.5,
            logprobs=None,
        )
        assert output.index == 0
        assert output.text == "Hello world"
        assert output.token_ids == [1, 2, 3]
        assert output.cumulative_logprob == -1.5
        assert output.logprobs is None

    def test_finished_true(self):
        output = CompletionOutput(
            index=0,
            text="done",
            token_ids=[1],
            cumulative_logprob=None,
            logprobs=None,
            finish_reason="stop",
        )
        assert output.finished() is True

    def test_finished_false(self):
        output = CompletionOutput(
            index=0,
            text="in progress",
            token_ids=[1],
            cumulative_logprob=None,
            logprobs=None,
        )
        assert output.finished() is False

    def test_repr(self):
        output = CompletionOutput(
            index=1,
            text="test",
            token_ids=[5, 6],
            cumulative_logprob=-0.5,
            logprobs=None,
            finish_reason="length",
            stop_reason="max_tokens",
        )
        repr_str = repr(output)
        assert "index=1" in repr_str
        assert "text='test'" in repr_str
        assert "token_ids=[5, 6]" in repr_str
        assert "finish_reason=length" in repr_str
        assert "stop_reason=max_tokens" in repr_str


# =============================================================================
# RequestOutput tests
# =============================================================================


class TestRequestOutput:
    def test_construction(self):
        output = RequestOutput(
            request_id="req-1",
            prompt="Hello",
            prompt_token_ids=[1, 2],
            prompt_logprobs=None,
            outputs=[],
            finished=False,
        )
        assert output.request_id == "req-1"
        assert output.prompt == "Hello"
        assert output.prompt_token_ids == [1, 2]
        assert output.finished is False
        assert output.outputs == []

    def test_forward_compatibility_with_kwargs(self):
        output = RequestOutput(
            request_id="req-1",
            prompt="test",
            prompt_token_ids=[1],
            prompt_logprobs=None,
            outputs=[],
            finished=False,
            future_param="some_value",
        )
        assert output.request_id == "req-1"

    def test_add_delta_aggregate(self):
        comp1 = CompletionOutput(
            index=0,
            text="Hello",
            token_ids=[1, 2],
            cumulative_logprob=-1.0,
            logprobs=None,
        )
        output = RequestOutput(
            request_id="req-1",
            prompt="test",
            prompt_token_ids=[1],
            prompt_logprobs=None,
            outputs=[comp1],
            finished=False,
        )
        comp2 = CompletionOutput(
            index=0,
            text=" world",
            token_ids=[3, 4],
            cumulative_logprob=-2.0,
            logprobs=None,
        )
        next_output = RequestOutput(
            request_id="req-1",
            prompt="test",
            prompt_token_ids=[1],
            prompt_logprobs=None,
            outputs=[comp2],
            finished=True,
        )
        output.add(next_output, aggregate=True)
        assert output.finished is True
        assert output.outputs[0].text == "Hello world"
        assert list(output.outputs[0].token_ids) == [1, 2, 3, 4]
        assert output.outputs[0].cumulative_logprob == -2.0

    def test_add_delta_replace(self):
        comp1 = CompletionOutput(
            index=0,
            text="old",
            token_ids=[1],
            cumulative_logprob=None,
            logprobs=None,
        )
        output = RequestOutput(
            request_id="req-1",
            prompt="test",
            prompt_token_ids=[1],
            prompt_logprobs=None,
            outputs=[comp1],
            finished=False,
        )
        comp2 = CompletionOutput(
            index=0,
            text="new",
            token_ids=[2, 3],
            cumulative_logprob=-1.0,
            logprobs=None,
        )
        next_output = RequestOutput(
            request_id="req-1",
            prompt="test",
            prompt_token_ids=[1],
            prompt_logprobs=None,
            outputs=[comp2],
            finished=False,
        )
        output.add(next_output, aggregate=False)
        assert output.outputs[0].text == "new"
        assert list(output.outputs[0].token_ids) == [2, 3]

    def test_add_new_completion_index(self):
        comp1 = CompletionOutput(
            index=0,
            text="first",
            token_ids=[1],
            cumulative_logprob=None,
            logprobs=None,
        )
        output = RequestOutput(
            request_id="req-1",
            prompt="test",
            prompt_token_ids=[1],
            prompt_logprobs=None,
            outputs=[comp1],
            finished=False,
        )
        comp2 = CompletionOutput(
            index=1,
            text="second",
            token_ids=[2],
            cumulative_logprob=None,
            logprobs=None,
        )
        next_output = RequestOutput(
            request_id="req-1",
            prompt="test",
            prompt_token_ids=[1],
            prompt_logprobs=None,
            outputs=[comp2],
            finished=False,
        )
        output.add(next_output, aggregate=True)
        assert len(output.outputs) == 2
        assert output.outputs[1].text == "second"

    def test_repr(self):
        output = RequestOutput(
            request_id="req-42",
            prompt="hi",
            prompt_token_ids=[1],
            prompt_logprobs=None,
            outputs=[],
            finished=True,
        )
        repr_str = repr(output)
        assert "req-42" in repr_str
        assert "finished=True" in repr_str


# =============================================================================
# PoolingOutput tests
# =============================================================================


class TestPoolingOutput:
    def test_construction(self):
        data = torch.randn(768)
        output = PoolingOutput(data=data)
        assert torch.equal(output.data, data)

    def test_repr(self):
        data = torch.randn(768)
        output = PoolingOutput(data=data)
        assert "PoolingOutput" in repr(output)

    def test_equality_same_data(self):
        data = torch.tensor([1.0, 2.0, 3.0])
        output1 = PoolingOutput(data=data.clone())
        output2 = PoolingOutput(data=data.clone())
        assert output1 == output2

    def test_equality_different_data(self):
        output1 = PoolingOutput(data=torch.tensor([1.0, 2.0]))
        output2 = PoolingOutput(data=torch.tensor([3.0, 4.0]))
        assert output1 != output2

    def test_equality_different_type(self):
        output = PoolingOutput(data=torch.tensor([1.0]))
        assert output != "not a pooling output"


# =============================================================================
# PoolingRequestOutput tests
# =============================================================================


class TestPoolingRequestOutput:
    def test_construction(self):
        data = torch.randn(768)
        pooling_output = PoolingOutput(data=data)
        output = PoolingRequestOutput(
            request_id="req-1",
            outputs=pooling_output,
            prompt_token_ids=[1, 2, 3],
            num_cached_tokens=0,
            finished=True,
        )
        assert output.request_id == "req-1"
        assert output.finished is True
        assert output.num_cached_tokens == 0

    def test_repr(self):
        data = torch.randn(4)
        pooling_output = PoolingOutput(data=data)
        output = PoolingRequestOutput(
            request_id="req-pool",
            outputs=pooling_output,
            prompt_token_ids=[1],
            num_cached_tokens=2,
            finished=True,
        )
        repr_str = repr(output)
        assert "req-pool" in repr_str
        assert "finished=True" in repr_str


# =============================================================================
# EmbeddingOutput tests
# =============================================================================


class TestEmbeddingOutput:
    def test_from_base(self):
        data = torch.tensor([0.1, 0.2, 0.3, 0.4])
        pooling_output = PoolingOutput(data=data)
        embedding = EmbeddingOutput.from_base(pooling_output)
        assert embedding.embedding == pytest.approx([0.1, 0.2, 0.3, 0.4], abs=1e-5)

    def test_hidden_size(self):
        embedding = EmbeddingOutput(embedding=[1.0, 2.0, 3.0])
        assert embedding.hidden_size == 3

    def test_from_base_non_1d_raises(self):
        data = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        pooling_output = PoolingOutput(data=data)
        with pytest.raises(ValueError, match="1-D embedding"):
            EmbeddingOutput.from_base(pooling_output)

    def test_repr(self):
        embedding = EmbeddingOutput(embedding=[1.0, 2.0, 3.0, 4.0])
        assert "hidden_size=4" in repr(embedding)


# =============================================================================
# EmbeddingRequestOutput tests
# =============================================================================


class TestEmbeddingRequestOutput:
    def test_from_base(self):
        data = torch.tensor([0.5, 0.6, 0.7])
        pooling_output = PoolingOutput(data=data)
        base_output = PoolingRequestOutput(
            request_id="req-emb",
            outputs=pooling_output,
            prompt_token_ids=[1, 2],
            num_cached_tokens=1,
            finished=True,
        )
        emb_output = EmbeddingRequestOutput.from_base(base_output)
        assert emb_output.request_id == "req-emb"
        assert isinstance(emb_output.outputs, EmbeddingOutput)
        assert emb_output.outputs.hidden_size == 3


# =============================================================================
# ClassificationOutput tests
# =============================================================================


class TestClassificationOutput:
    def test_from_base(self):
        data = torch.tensor([0.1, 0.7, 0.2])
        pooling_output = PoolingOutput(data=data)
        classification = ClassificationOutput.from_base(pooling_output)
        assert len(classification.probs) == 3

    def test_num_classes(self):
        classification = ClassificationOutput(probs=[0.3, 0.7])
        assert classification.num_classes == 2

    def test_from_base_non_1d_raises(self):
        data = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        pooling_output = PoolingOutput(data=data)
        with pytest.raises(ValueError, match="1-D probability"):
            ClassificationOutput.from_base(pooling_output)

    def test_repr(self):
        classification = ClassificationOutput(probs=[0.5, 0.3, 0.2])
        assert "num_classes=3" in repr(classification)


# =============================================================================
# ClassificationRequestOutput tests
# =============================================================================


class TestClassificationRequestOutput:
    def test_from_base(self):
        data = torch.tensor([0.8, 0.2])
        pooling_output = PoolingOutput(data=data)
        base_output = PoolingRequestOutput(
            request_id="req-cls",
            outputs=pooling_output,
            prompt_token_ids=[1],
            num_cached_tokens=0,
            finished=True,
        )
        cls_output = ClassificationRequestOutput.from_base(base_output)
        assert cls_output.request_id == "req-cls"
        assert isinstance(cls_output.outputs, ClassificationOutput)
        assert cls_output.outputs.num_classes == 2


# =============================================================================
# ScoringOutput tests
# =============================================================================


class TestScoringOutput:
    def test_from_base_scalar(self):
        data = torch.tensor(0.85)
        pooling_output = PoolingOutput(data=data)
        scoring = ScoringOutput.from_base(pooling_output)
        assert scoring.score == pytest.approx(0.85, abs=1e-5)

    def test_from_base_1d_single(self):
        data = torch.tensor([0.42])
        pooling_output = PoolingOutput(data=data)
        scoring = ScoringOutput.from_base(pooling_output)
        assert scoring.score == pytest.approx(0.42, abs=1e-5)

    def test_from_base_non_scalar_raises(self):
        data = torch.tensor([0.1, 0.2])
        pooling_output = PoolingOutput(data=data)
        with pytest.raises(ValueError, match="scalar score"):
            ScoringOutput.from_base(pooling_output)

    def test_repr(self):
        scoring = ScoringOutput(score=0.95)
        assert "score=0.95" in repr(scoring)


# =============================================================================
# ScoringRequestOutput tests
# =============================================================================


class TestScoringRequestOutput:
    def test_from_base(self):
        data = torch.tensor(0.75)
        pooling_output = PoolingOutput(data=data)
        base_output = PoolingRequestOutput(
            request_id="req-score",
            outputs=pooling_output,
            prompt_token_ids=[1, 2],
            num_cached_tokens=0,
            finished=True,
        )
        score_output = ScoringRequestOutput.from_base(base_output)
        assert score_output.request_id == "req-score"
        assert isinstance(score_output.outputs, ScoringOutput)
        assert score_output.outputs.score == pytest.approx(0.75, abs=1e-5)

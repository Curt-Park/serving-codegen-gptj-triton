"""Locust file for load tests.

Reference:
    http://docs.locust.io/en/stable/writing-a-locustfile.html
    http://docs.locust.io/en/stable/increase-performance.html
    http://docs.locust.io/en/stable/running-distributed.html
"""
from functools import lru_cache
from typing import Any, List

import numpy as np
import tritonclient.http as httpclient
from locust import FastHttpUser, task
from tritonclient.utils import np_to_triton_dtype


class APIUser(FastHttpUser):
    """Send api user requests."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize."""
        super().__init__(*args, **kwargs)

    @task
    def request(self) -> None:
        """Request model inference."""
        request_body, json_size = httpclient._get_inference_request(
            inputs=self._prepare_inputs(),
            request_id="",
            outputs=None,
            sequence_id=0,
            sequence_start=False,
            sequence_end=False,
            priority=0,
            timeout=None,
        )
        self.client.post(
            "/v2/models/ensemble/versions/1/infer",
            data=request_body,
            headers={
                "Inference-Header-Content-Length": json_size,
            },
        )

    @lru_cache
    def _prepare_inputs(self) -> List[Any]:
        """Prepare triton inputs."""
        input0 = [["def helloworld():"]]
        input0_data = np.array(input0).astype(object)
        output0_len = 128 * np.ones_like(input0).astype(np.uint32)
        bad_words_list = np.array([[""]], dtype=object)
        stop_words_list = np.array([[""]], dtype=object)
        runtime_top_k = 3 * np.ones([1, 1]).astype(np.uint32)
        runtime_top_p = 0.92 * np.ones([1, 1]).astype(np.float32)
        temperature = 0.5 * np.ones([1, 1]).astype(np.float32)
        len_penalty = -1 * np.ones([1, 1]).astype(np.float32)
        repetition_penalty = 1.1 * np.ones([1, 1]).astype(np.float32)
        random_seed = 42 * np.ones([1, 1]).astype(np.uint64)
        is_return_log_probs = np.ones([1, 1]).astype(bool)
        beam_width = np.ones([1, 1]).astype(np.uint32)
        beam_diversity = 0.5 * np.ones([1, 1]).astype(np.float32)
        start_ids = 220 * np.ones([1, 1]).astype(np.uint32)
        end_ids = 50256 * np.ones([1, 1]).astype(np.uint32)

        inputs = [
            prepare_tensor("INPUT_0", input0_data),
            prepare_tensor("INPUT_1", output0_len),
            prepare_tensor("INPUT_2", bad_words_list),
            prepare_tensor("INPUT_3", stop_words_list),
            prepare_tensor("runtime_top_k", runtime_top_k),
            prepare_tensor("runtime_top_p", runtime_top_p),
            prepare_tensor("temperature", temperature),
            prepare_tensor("len_penalty", len_penalty),
            prepare_tensor("repetition_penalty", repetition_penalty),
            prepare_tensor("random_seed", random_seed),
            prepare_tensor("is_return_log_probs", is_return_log_probs),
            prepare_tensor("beam_width", beam_width),
            prepare_tensor("beam_search_diversity_rate", beam_diversity),
            prepare_tensor("start_id", start_ids),
            prepare_tensor("end_id", end_ids),
        ]

        return inputs


def prepare_tensor(name: str, tensor: np.ndarray) -> httpclient.InferInput:
    """Create a triton input."""
    infer_input = httpclient.InferInput(
        name, tensor.shape, np_to_triton_dtype(tensor.dtype)
    )
    infer_input.set_data_from_numpy(tensor)
    return infer_input

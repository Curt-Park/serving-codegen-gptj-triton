"""CLI Example for CodeGen-350M.

- Author: Curt Park
- Email: www.jwpark.co.kr@gmail.com
"""
import os

import gradio as gr
import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype

example = [
    ["def print_hello_world():", 1024, 3, 0.92, 0.5, 1.0, -1, 5, 42, 1],
    ["def get_file_size(filepath):", 1024, 3, 0.92, 0.5, 1.0, -1, 1, 42, 1],
    ["def count_lines(filename):", 1024, 3, 0.92, 0.5, 1.0, -1, 1.16, 42, 1],
    ["def count_words(filename):", 1024, 3, 0.92, 0.5, 0.3, -1, 1, 42, 1],
    ["def two_sum(nums, target):", 1024, 3, 0.92, 0.5, 1.0, -1, 1, 60, 1],
    [
        "Solve the two sum problem with hash map.",
        1024,
        3,
        0.92,
        0.5,
        0.1,
        -1,
        1,
        205,
        1,
    ],
]


URL = os.getenv("TRITON_SERVER_URL", "localhost:8001")
tritonclient = grpcclient.InferenceServerClient(URL)


def prepare_tensor(name: str, tensor: np.ndarray) -> grpcclient.InferInput:
    """Create a triton input."""
    infer_input = grpcclient.InferInput(
        name, tensor.shape, np_to_triton_dtype(tensor.dtype)
    )
    infer_input.set_data_from_numpy(tensor)
    return infer_input


# pylint: disable=too-many-locals,broad-except
def code_generation(
    gen_prompt: str,
    max_tokens: int,
    top_k: int,
    top_p: float,
    diversity: float,
    temp: float,
    len_penalty_: float,
    rep_penalty: float,
    seed: int,
    beams: int,
) -> str:
    """Generate code."""
    input0 = [[gen_prompt]]
    input0_data = np.array(input0).astype(object)
    output0_len = np.ones_like(input0).astype(np.uint32) * max_tokens
    bad_words_list = np.array([[""]], dtype=object)
    stop_words_list = np.array([[""]], dtype=object)
    runtime_top_k = top_k * np.ones([1, 1]).astype(np.uint32)
    runtime_top_p = top_p * np.ones([1, 1]).astype(np.float32)
    temperature = temp * np.ones([1, 1]).astype(np.float32)
    len_penalty = len_penalty_ * np.ones([1, 1]).astype(np.float32)
    repetition_penalty = rep_penalty * np.ones([1, 1]).astype(np.float32)
    random_seed = seed * np.ones([1, 1]).astype(np.uint64)
    is_return_log_probs = np.ones([1, 1]).astype(bool)
    beam_width = beams * np.ones([1, 1]).astype(np.uint32)
    beam_diversity = diversity * np.ones([1, 1]).astype(np.float32)
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

    try:
        result = tritonclient.infer("ensemble", inputs)
        output0 = result.as_numpy("OUTPUT_0")
    except Exception as exception:
        print(exception)

    resp = output0[0].decode("utf-8")
    idx = resp.find("<|endoftext|>")
    return resp[:idx]


gr.Interface(
    fn=code_generation,
    inputs=[
        gr.Code(lines=10, label="Input code"),
        gr.Slider(
            minimum=8,
            maximum=1024,
            step=1,
            value=1024,
            label="Number of tokens to generate",
        ),
        gr.Slider(
            minimum=1,
            maximum=100,
            step=1,
            value=5,
            label="Top K",
        ),
        gr.Slider(
            minimum=0,
            maximum=1,
            step=0.01,
            value=0.92,
            label="Top P",
        ),
        gr.Slider(
            minimum=0,
            maximum=1,
            step=0.01,
            value=0.5,
            label="Beam Search Diversity",
        ),
        gr.Slider(
            minimum=0,
            maximum=2.5,
            step=0.1,
            value=0.6,
            label="Temperature",
        ),
        gr.Slider(
            minimum=-1,
            maximum=1,
            step=0.01,
            value=0.0,
            label="Length Penalty",
        ),
        gr.Slider(
            minimum=0,
            maximum=5,
            step=0.01,
            value=1.0,
            label="Repetition Penalty",
        ),
        gr.Slider(
            minimum=0,
            maximum=1000,
            step=1,
            value=42,
            label="Random seed to use for the generation",
        ),
        gr.Slider(
            minimum=1,
            maximum=100,
            step=1,
            value=1,
            label="Beams",
        ),
    ],
    outputs=gr.Code(label="Predicted code", lines=10),
    allow_flagging="never",
    examples=example,
    title="CodeGen Generator",
).queue().launch()

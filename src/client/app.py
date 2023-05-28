"""CLI Example for CodeGen-350M.

- Author: Curt Park
- Email: www.jwpark.co.kr@gmail.com
"""
import os
import gradio as gr
import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype

example = [
    ["def print_hello_world():", 8, 0.6, 42],
    ["def get_file_size(filepath):", 27, 0.6, 42],
    ["def count_lines(filename):", 35, 0.6, 42],
    ["def count_words(filename):", 42, 0.6, 42],
    ["def two_sum(nums, target):", 140, 0.6, 55],
]


URL = os.getenv("TRITON_SERVER_URL", "localhost:8000")
tritonclient = httpclient.InferenceServerClient(URL, concurrency=1)


def prepare_tensor(name: str, tensor: np.ndarray) -> httpclient.InferInput:
    """Create a triton input."""
    infer_input = httpclient.InferInput(
        name, tensor.shape, np_to_triton_dtype(tensor.dtype)
    )
    infer_input.set_data_from_numpy(tensor)
    return infer_input


# pylint: disable=too-many-locals,broad-except
def code_generation(
    gen_prompt: str, max_tokens: int, temp: float = 0.6, seed: int = 42
) -> str:
    """Generate code."""
    input0 = [[gen_prompt]]
    input0_data = np.array(input0).astype(object)
    output0_len = np.ones_like(input0).astype(np.uint32) * max_tokens
    bad_words_list = np.array([[""]], dtype=object)
    stop_words_list = np.array([[""]], dtype=object)
    runtime_top_k = np.ones([input0_data.shape[0], 1]).astype(np.uint32)
    runtime_top_p = np.zeros([input0_data.shape[0], 1]).astype(np.float32)
    beam_search_diversity_rate = np.zeros([input0_data.shape[0], 1]).astype(np.float32)
    temperature = temp * np.ones([input0_data.shape[0], 1]).astype(np.float32)
    len_penalty = 1.0 * np.ones([input0_data.shape[0], 1]).astype(np.float32)
    repetition_penalty = 1.0 * np.ones([input0_data.shape[0], 1]).astype(np.float32)
    random_seed = seed * np.ones([input0_data.shape[0], 1]).astype(np.uint64)
    is_return_log_probs = np.ones([input0_data.shape[0], 1]).astype(bool)
    beam_width = np.ones([input0_data.shape[0], 1]).astype(np.uint32)
    start_ids = 220 * np.ones([input0_data.shape[0], 1]).astype(np.uint32)
    end_ids = 50256 * np.ones([input0_data.shape[0], 1]).astype(np.uint32)

    inputs = [
        prepare_tensor("INPUT_0", input0_data),
        prepare_tensor("INPUT_1", output0_len),
        prepare_tensor("INPUT_2", bad_words_list),
        prepare_tensor("INPUT_3", stop_words_list),
        prepare_tensor("runtime_top_k", runtime_top_k),
        prepare_tensor("runtime_top_p", runtime_top_p),
        prepare_tensor("beam_search_diversity_rate", beam_search_diversity_rate),
        prepare_tensor("temperature", temperature),
        prepare_tensor("len_penalty", len_penalty),
        prepare_tensor("repetition_penalty", repetition_penalty),
        prepare_tensor("random_seed", random_seed),
        prepare_tensor("is_return_log_probs", is_return_log_probs),
        prepare_tensor("beam_width", beam_width),
        prepare_tensor("start_id", start_ids),
        prepare_tensor("end_id", end_ids),
    ]

    try:
        result = tritonclient.infer("ensemble", inputs)
        output0 = result.as_numpy("OUTPUT_0")
    except Exception as exception:
        print(exception)
    return output0[0].decode("utf-8").replace("<|endoftext|>", "\t")


gr.Interface(
    fn=code_generation,
    inputs=[
        gr.Code(lines=10, label="Input code"),
        gr.Slider(
            minimum=8,
            maximum=1000,
            step=1,
            value=8,
            label="Number of tokens to generate",
        ),
        gr.Slider(
            minimum=0,
            maximum=2.5,
            step=0.1,
            value=0.6,
            label="Temperature",
        ),
        gr.Slider(
            minimum=0,
            maximum=1000,
            step=1,
            value=42,
            label="Random seed to use for the generation",
        ),
    ],
    outputs=gr.Code(label="Predicted code", lines=10),
    allow_flagging="never",
    examples=example,
    title="CodeGen Generator",
).queue().launch()

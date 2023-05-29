"""A simple client for CodeGen-350M.

- Author: Curt Park
- Email: www.jwpark.co.kr@gmail.com
"""
import os
from typing import List, Tuple

import gradio as gr
import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype

URL = os.getenv("TRITON_SERVER_URL", "localhost:8001")
tritonclient = grpcclient.InferenceServerClient(URL)


def add_text(history: List[Tuple[str, str]], text: str) -> List[str]:
    """Add the input text."""
    history += [(text, None)]
    return history


# pylint: disable=too-many-locals,broad-except
def bot(
    history: List[Tuple[str]],
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
    """Predict."""
    gen_prompt = history[-1][0]
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
    resp = resp[:idx]
    history[-1][1] = "```\n" + resp + """\n```"""

    return history


def prepare_tensor(name: str, tensor: np.ndarray) -> grpcclient.InferInput:
    """Create a triton input."""
    infer_input = grpcclient.InferInput(
        name, tensor.shape, np_to_triton_dtype(tensor.dtype)
    )
    infer_input.set_data_from_numpy(tensor)
    return infer_input


with gr.Blocks() as demo:
    chatbot = gr.Chatbot([], elem_id="chatbot").style(height=750)

    with gr.Row():
        txt = gr.Textbox(
            show_label=False,
            placeholder="Enter text and press enter",
        ).style(container=False)

    with gr.Row():
        n_tokens = gr.Slider(
            minimum=8,
            maximum=1024,
            step=1,
            value=256,
            label="Number of tokens to generate",
        )
        top_k = gr.Slider(
            minimum=1,
            maximum=100,
            step=1,
            value=3,
            label="Top K",
        )
        top_p = gr.Slider(
            minimum=0,
            maximum=1,
            step=0.01,
            value=0.92,
            label="Top P",
        )
        beam_diversity = gr.Slider(
            minimum=0,
            maximum=1,
            step=0.01,
            value=0.5,
            label="Beam Search Diversity",
        )
        temperature = gr.Slider(
            minimum=0,
            maximum=2.5,
            step=0.1,
            value=0.6,
            label="Temperature",
        )
        len_penalty = gr.Slider(
            minimum=-1,
            maximum=1,
            step=0.01,
            value=0.0,
            label="Length Penalty",
        )
        rep_penalty = gr.Slider(
            minimum=0,
            maximum=5,
            step=0.01,
            value=1.0,
            label="Repetition Penalty",
        )
        seed = gr.Slider(
            minimum=0,
            maximum=1000,
            step=1,
            value=128,
            label="Random seed to use for the generation",
        )
        beams = gr.Slider(
            minimum=1,
            maximum=100,
            step=1,
            value=1,
            label="Beams",
        )

    txt.submit(add_text, [chatbot, txt], chatbot,).then(
        bot,
        [
            chatbot,
            n_tokens,
            top_k,
            top_p,
            beam_diversity,
            temperature,
            len_penalty,
            rep_penalty,
            seed,
            beams,
        ],
        chatbot,
    )

demo.queue().launch()

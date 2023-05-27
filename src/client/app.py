"""CLI Example for CodeGen-350M.

- Author: Curt Park
- Email: www.jwpark.co.kr@gmail.com
"""
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed

example = [
    ["def print_hello_world():", 8, 0.6, 42],
    ["def get_file_size(filepath):", 27, 0.6, 42],
    ["def count_lines(filename):", 35, 0.6, 42],
    ["def count_words(filename):", 42, 0.6, 42],
    ["def two_sum(nums, target):", 140, 0.6, 55],
]
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
model = AutoModelForCausalLM.from_pretrained("moyix/codegen-350M-mono-gptj")


def code_generation(
    gen_prompt, max_tokens, temperature=0.6, seed=42  # noqa: ANN001
) -> str:
    """Generate code."""
    set_seed(seed)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    generated_text = pipe(
        gen_prompt,
        do_sample=True,
        top_p=0.95,
        temperature=temperature,
        max_new_tokens=max_tokens,
    )[0]["generated_text"]
    return generated_text


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

import argparse
import os
from functools import lru_cache
from types import SimpleNamespace
from typing import Optional, Tuple

import gradio as gr
import torch

from .cli import ASPECT_RATIOS, create_generator, load_pipeline, pick_device
from .lora_utils import load_lora_weights


def _coerce_int(value: Optional[int], default: int) -> int:
    try:
        v = int(value)
        return v if v > 0 else default
    except Exception:
        return default


def get_available_loras():
    """Get list of available LoRA directories."""
    loras = ["None"]
    loras_dir = "loras"
    if os.path.exists(loras_dir):
        for item in os.listdir(loras_dir):
            if os.path.isdir(os.path.join(loras_dir, item)):
                loras.append(item)
    return loras


@lru_cache(maxsize=1)
def _cached_pipeline(
    device_choice: str,
    attention_backend: str,
    compile_flag: bool,
    cpu_offload: bool,
) -> Tuple:
    device, dtype = pick_device(device_choice)
    dummy_args = SimpleNamespace(
        attention_backend=attention_backend,
        compile=compile_flag,
        cpu_offload=cpu_offload,
    )
    pipe = load_pipeline(dummy_args, device, dtype)
    return pipe, device, dtype


def _resolve_lora_path(lora_name: str) -> Optional[str]:
    """Match CLI behavior: resolve a LoRA directory/name to a .safetensors file."""
    if not lora_name or lora_name == "None":
        return None
    lora_path = os.path.join("loras", lora_name)
    if not os.path.exists(lora_path):
        return None
    if os.path.isdir(lora_path):
        safetensors_files = [f for f in os.listdir(lora_path) if f.endswith(".safetensors")]
        return os.path.join(lora_path, safetensors_files[0]) if safetensors_files else None
    if lora_path.endswith(".safetensors"):
        return lora_path
    return None


def _ensure_lora(pipe, lora_name: str, lora_scale: float):
    """
    Reuse the loaded pipeline while swapping LoRAs:
    - unload previous merge if present
    - load requested LoRA if provided
    """
    merger = getattr(pipe, "_lora_merger", None)

    if merger:
        merger.unload_lora_weights()
        pipe._lora_merger = None

    lora_path = _resolve_lora_path(lora_name)
    if lora_path:
        pipe._lora_merger = load_lora_weights(pipe, lora_path, lora_scale)
        print(f"Loaded LoRA from {lora_path} with scale {lora_scale}")

    return pipe


def generate_image(
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance: float,
    aspect: str,
    height: int,
    width: int,
    seed: Optional[int],
    attention_backend: str,
    device_choice: str,
    compile_flag: bool,
    cpu_offload: bool,
    lora_name: str,
    lora_scale: float,
):
    steps = max(1, int(steps))
    guidance = float(guidance)
    if aspect != "custom":
        h, w = ASPECT_RATIOS.get(aspect, (1024, 1024))
    else:
        h = _coerce_int(height, 1024)
        w = _coerce_int(width, 1024)

    pipe, device, dtype = _cached_pipeline(device_choice, attention_backend, compile_flag, cpu_offload)
    pipe = _ensure_lora(pipe, lora_name, lora_scale)

    if seed is None or seed == 0:
        seed = torch.seed() % (2**63 - 1)
    else:
        seed = int(seed)

    generator = create_generator(device, seed)

    lora_info = f", lora={lora_name}, lora_scale={lora_scale}" if lora_name != "None" else ""
    info = (
        f"device={device}, dtype={dtype}, steps={steps}, "
        f"guidance={guidance}, size={w}x{h}, seed={seed}, "
        f"attn={attention_backend}, compile={compile_flag}, "
        f"cpu_offload={cpu_offload}{lora_info}"
    )

    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            height=h,
            width=w,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        )

    return result.images[0], info


def build_app():
    with gr.Blocks(title="Z-Image Turbo") as demo:
        gr.Markdown("# Z-Image Turbo (MPS/CUDA/CPU)")

        with gr.Row():
            with gr.Column(scale=3):
                prompt = gr.Textbox(
                    label="Prompt",
                    value="Analog film portrait of a skateboarder, shallow depth of field",
                    lines=3,
                )
                negative = gr.Textbox(label="Negative prompt", value="", lines=2)

                with gr.Row():
                    steps = gr.Slider(
                        label="Steps",
                        minimum=1,
                        maximum=20,
                        value=9,
                        step=1,
                    )
                    guidance = gr.Slider(
                        label="Guidance scale (Turbo uses 0.0)",
                        minimum=0.0,
                        maximum=5.0,
                        value=0.0,
                        step=0.1,
                    )

                aspect = gr.Dropdown(
                    label="Aspect ratio",
                    choices=list(ASPECT_RATIOS.keys()) + ["custom"],
                    value="1:1",
                )
                with gr.Row():
                    height = gr.Number(label="Height (px, custom)", value=1024, precision=0)
                    width = gr.Number(label="Width (px, custom)", value=1024, precision=0)

                seed = gr.Number(label="Seed (0 or empty = random)", value=0, precision=0)

            with gr.Column(scale=2):
                device_choice = gr.Radio(
                    label="Device",
                    choices=["auto", "mps", "cuda", "cpu"],
                    value="auto",
                    interactive=True,
                )
                attention_backend = gr.Radio(
                    label="Attention backend",
                    choices=["sdpa", "flash2", "flash3"],
                    value="sdpa",
                )
                compile_flag = gr.Checkbox(label="torch.compile DiT (CUDA best)", value=False)
                cpu_offload = gr.Checkbox(label="CPU offload (CUDA only)", value=False)

                lora_name = gr.Dropdown(
                    label="LoRA",
                    choices=get_available_loras(),
                    value="None",
                )
                lora_scale = gr.Slider(
                    label="LoRA Scale",
                    minimum=0.0,
                    maximum=2.0,
                    value=1.3,
                    step=0.1,
                )

                run_btn = gr.Button("Generate", variant="primary")

        with gr.Row():
            image_out = gr.Image(label="Result")
            info = gr.Textbox(label="Run info", interactive=False)

        run_btn.click(
            fn=generate_image,
            inputs=[
                prompt,
                negative,
                steps,
                guidance,
                aspect,
                height,
                width,
                seed,
                attention_backend,
                device_choice,
                compile_flag,
                cpu_offload,
                lora_name,
                lora_scale,
            ],
            outputs=[image_out, info],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(
        description="Launch a Gradio demo for Z-Image-Turbo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind.")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind.")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share link.")
    args = parser.parse_args()

    demo = build_app()
    demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

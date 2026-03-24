import base64
import time
from io import BytesIO
from typing import List

import streamlit as st
from openai import OpenAI
from PIL import Image


DEFAULT_BASE_URL = "http://0.0.0.0:8989/v1"
DEFAULT_API_KEY = "EMPTY"
DEFAULT_MODEL = "qwen_2_5_vl_7b"
MAX_IMAGES = 2


def pil_to_data_url(image: Image.Image) -> str:
    """Convert PIL image to base64 data URL for OpenAI-compatible vision API."""
    buffered = BytesIO()
    rgb_image = image.convert("RGB")
    rgb_image.save(buffered, format="JPEG", quality=90)
    b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def build_user_content(prompt: str, images: List[Image.Image]) -> List[dict]:
    content = [{"type": "text", "text": prompt}]
    for img in images:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": pil_to_data_url(img)},
            }
        )
    return content


def main() -> None:
    st.set_page_config(page_title="vLLM Vision Prompt Tester", layout="wide")
    st.title("vLLM Vision Prompt Tester")
    st.caption("Upload image(s) and send prompt to your vLLM service.")

    with st.sidebar:
        st.subheader("vLLM Config")
        base_url = st.text_input("Base URL", value=DEFAULT_BASE_URL)
        api_key = st.text_input("API Key", value=DEFAULT_API_KEY)
        model = st.text_input("Model Name", value=DEFAULT_MODEL)
        max_tokens = st.number_input("Max tokens", min_value=1, max_value=4096, value=512, step=1)
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.5, value=0.0, step=0.1)
        timeout_sec = st.number_input("Timeout (seconds)", min_value=5, max_value=300, value=90, step=5)

    uploaded_files = st.file_uploader(
        f"Upload image(s), up to {MAX_IMAGES}",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
    )
    prompt = st.text_area("Prompt", value="Describe this image.", height=140)

    images: List[Image.Image] = []
    if uploaded_files:
        if len(uploaded_files) > MAX_IMAGES:
            st.error(f"You uploaded {len(uploaded_files)} images, but max is {MAX_IMAGES}.")
            return
        preview_cols = st.columns(len(uploaded_files))
        for i, f in enumerate(uploaded_files):
            img = Image.open(f)
            images.append(img)
            with preview_cols[i]:
                st.image(img, caption=f.name, use_container_width=True)

    send_btn = st.button("Send to vLLM", type="primary", use_container_width=True)

    if send_btn:
        if not prompt.strip():
            st.warning("Prompt cannot be empty.")
            return
        if not images:
            st.warning("Please upload at least one image.")
            return

        try:
            client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_sec)
            messages = [{"role": "user", "content": build_user_content(prompt.strip(), images)}]

            with st.spinner("Requesting vLLM..."):
                t0 = time.time()
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=int(max_tokens),
                    temperature=float(temperature),
                )
                elapsed = time.time() - t0

            answer = response.choices[0].message.content if response.choices else ""
            st.success(f"Done in {elapsed:.2f}s")
            st.subheader("Model Output")
            st.write(answer)

            if getattr(response, "usage", None) is not None:
                st.caption(
                    f"Tokens - prompt: {response.usage.prompt_tokens}, "
                    f"completion: {response.usage.completion_tokens}, "
                    f"total: {response.usage.total_tokens}"
                )

            with st.expander("Raw response object"):
                st.json(response.model_dump())

        except Exception as e:
            st.error(f"Request failed: {e}")


if __name__ == "__main__":
    main()

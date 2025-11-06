import streamlit as st
import requests
import io
import json
import base64
import html  

try:
    from PIL import Image
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    TESSERACT_AVAILABLE = True
except Exception:
    TESSERACT_AVAILABLE = False

# ---------------- HTML ESCAPE HELPER ----------------
def escape_html(text: str) -> str:
    """Safely escape HTML special characters."""
    return html.escape(text or "")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="ChatGPT", page_icon="☯", layout="wide")

# ---------------- SESSION STATE ----------------
if "chats" not in st.session_state:
    st.session_state.chats = []  
if "current_chat" not in st.session_state:
    st.session_state.current_chat = {"name": None, "messages": []}
if "chat_saved" not in st.session_state:
    st.session_state.chat_saved = False
if "menu_open" not in st.session_state:
    st.session_state.menu_open = {}  
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "codellama:7b"

# OCR staging
if "uploaded_image_bytes" not in st.session_state:
    st.session_state.uploaded_image_bytes = None
if "ocr_text" not in st.session_state:
    st.session_state.ocr_text = ""

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("<h1 style='font-family:Bookman Old Style; font-style:italic;'>Chats</h1>", unsafe_allow_html=True)
    st.markdown("---")

    # 1. MODEL SETTINGS
    st.subheader("🤖 Model Settings")
    model_choice = st.selectbox(
        "Choose a Model",
        ["codellama:7b", "mistral:7b", "llama2:7b"],
        index=0
    )

    model_descriptions = {
        "codellama:7b": "Best for *code explanation and generation*, (debugging, syntax).",
        "mistral:7b": "Fast and efficient for *general Q&A*, (reasoning, summaries).",
        "llama2:7b": "Balanced model for *general explanations and code help*."
    }
    st.info(model_descriptions[model_choice])

    st.markdown("---")

    # 2. NEW CHAT
    if st.button("New Chat"):
        st.session_state.current_chat = {"name": None, "messages": []}
        st.session_state.chat_saved = False
        st.session_state.uploaded_image_bytes = None
        st.session_state.ocr_text = ""
        st.rerun()

    st.markdown("---")

    # 3. SEARCH
    search_query = st.text_input("🔍 Search chats")

    st.markdown("---")

    # 4. CHAT HISTORY
    if search_query:
        filtered = [chat for chat in st.session_state.chats
                    if chat["name"] and search_query.lower() in chat["name"].lower()]
        st.write(f"{len(filtered)} out of {len(st.session_state.chats)} chats found")
    else:
        filtered = st.session_state.chats

    st.write("*Chat History*")
    for idx, chat in enumerate(reversed(filtered)):
        chat_name = chat["name"] if chat["name"] else "Untitled"
        display_name = chat_name if len(chat_name) <= 25 else chat_name[:25] + "..."

        col1, col2 = st.columns([8, 1])
        with col1:
            if st.button(display_name, key=f"chat_button_{idx}"):
                st.session_state.current_chat = {
                    "name": chat_name,
                    "messages": chat["messages"].copy()
                }
                st.session_state.chat_saved = True
                st.rerun()
        with col2:
            if st.button("⋯", key=f"menu_btn_{idx}"):
                st.session_state.menu_open[idx] = not st.session_state.menu_open.get(idx, False)

        if st.session_state.menu_open.get(idx, False):
            dcol1, dcol2 = st.columns(2)
            with dcol1:
                txt_content = ""
                for m in chat["messages"]:
                    txt_content += f"{m['role'].upper()}: {m['content']}\n\n"
                file = io.BytesIO(txt_content.encode("utf-8"))
                st.download_button(
                    label="Download",
                    data=file,
                    file_name=f"{chat_name[:20]}.txt",
                    mime="text/plain",
                    key=f"download_{idx}"
                )
            with dcol2:
                if st.button("Delete", key=f"delete_{idx}"):
                    real_index = len(filtered) - 1 - idx
                    del st.session_state.chats[real_index]
                    st.rerun()

# ---------------- TITLE ----------------
st.markdown("<h1 style='font-family:Georgia;'>Code GENI AI Explainer & Generator</h1>", unsafe_allow_html=True)

# ---------------- CHAT DISPLAY CONTAINER ----------------
chat_display = st.container()
with chat_display:
    for msg in st.session_state.current_chat["messages"]:
        if msg["role"] == "user":
            cols = st.columns([0.2, 0.8])
            with cols[1]:
                st.markdown(
                    f"<div style='background-color:#E5F1E5;padding:8px;border-radius:10px;margin:5px 0;"
                    f"float:right;display:inline-block;text-align:left;max-width:80%'>{msg['content']}</div>",
                    unsafe_allow_html=True,
                )
        else:
            cols = st.columns([0.8, 0.2])
            with cols[0]:
                st.markdown(
                    f"<div style='background-color:#F1F0F0;padding:8px;border-radius:10px;margin:5px 0;"
                    f"float:left;display:inline-block;text-align:left;max-width:80%'>{escape_html(msg['content'])}</div>",
                    unsafe_allow_html=True,
                )

# Placeholder for streaming assistant messages (above uploader)
streaming_placeholder = chat_display.empty()

# ---------------- OCR FILE UPLOADER ----------------
st.markdown("---")
uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB") if TESSERACT_AVAILABLE else None
    except Exception:
        image = None

    buf = io.BytesIO()
    if image:
        image.save(buf, format="PNG")
        st.session_state.uploaded_image_bytes = buf.getvalue()
    else:
        uploaded_file.seek(0)
        st.session_state.uploaded_image_bytes = uploaded_file.read()

    if TESSERACT_AVAILABLE and image:
        try:
            st.session_state.ocr_text = pytesseract.image_to_string(image).strip()
            st.success("OCR completed (text will be used internally).")
        except Exception as e:
            st.session_state.ocr_text = ""
            st.warning(f"OCR failed: {e}")
    else:
        st.session_state.ocr_text = ""
        if not TESSERACT_AVAILABLE:
            st.warning("pytesseract or PIL not installed — OCR disabled.")

# ---------------- CHAT INPUT ----------------
prompt = st.chat_input("Type your message...")

if prompt is not None:
    parts = []
    if prompt.strip():
        parts.append(f"<div style='margin-bottom:8px;font-weight:600;'>{escape_html(prompt)}</div>")

    if st.session_state.uploaded_image_bytes:
        b64 = base64.b64encode(st.session_state.uploaded_image_bytes).decode("utf-8")
        img_tag = f"<div style='margin:8px 0;'><img src='data:image/png;base64,{b64}' style='max-width:240px;border-radius:6px;display:block;' /></div>"
        parts.append(img_tag)

    if parts:
        user_content_html = "".join(parts)
        if not st.session_state.current_chat["name"]:
            st.session_state.current_chat["name"] = prompt.strip() or "Image Chat"

        st.session_state.current_chat["messages"].append({"role": "user", "content": user_content_html})

        if not st.session_state.chat_saved:
            st.session_state.chats.append(st.session_state.current_chat)
            st.session_state.chat_saved = True

        combined_prompt_parts = []
        if prompt.strip():
            combined_prompt_parts.append(f"User question: {prompt.strip()}")
        if st.session_state.ocr_text:
            combined_prompt_parts.append("Extracted text from uploaded image:")
            combined_prompt_parts.append(st.session_state.ocr_text)
        combined_prompt = "\n\n".join(combined_prompt_parts)

        st.session_state.uploaded_image_bytes = None
        st.session_state.ocr_text = ""
        st.session_state.last_combined_prompt = combined_prompt
        st.rerun()

# ---------------- ASSISTANT STREAMING ----------------
if st.session_state.current_chat["messages"] and st.session_state.current_chat["messages"][-1]["role"] == "user":
    import re
    last_user_html = st.session_state.current_chat["messages"][-1]["content"]
    try:
        user_prompt = st.session_state.get("last_combined_prompt", "")
    except Exception:
        user_prompt = last_user_html

    accumulated_text = ""
    try:
        with requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model_choice, "prompt": user_prompt, "stream": True, "options": {"num_predict": 2048}},
            stream=True
        ) as response:
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line.decode("utf-8"))
                            if "response" in data:
                                accumulated_text += data["response"]
                                streaming_placeholder.markdown(
                                    f"<div style='background-color:#F1F0F0;padding:8px;border-radius:10px;margin:5px 0;"
                                    f"float:left;display:inline-block;text-align:left;max-width:80%'>{escape_html(accumulated_text)}</div>",
                                    unsafe_allow_html=True
                                )
                        except Exception:
                            continue
            else:
                accumulated_text = f"⚠ Error {response.status_code}: {response.text}"
    except Exception as e:
        accumulated_text = f"⚠ Failed to connect to Ollama: {str(e)}"

    st.session_state.current_chat["messages"].append({"role": "assistant", "content": accumulated_text})
    st.rerun()

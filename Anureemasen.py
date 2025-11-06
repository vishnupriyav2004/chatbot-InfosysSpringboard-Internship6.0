import streamlit as st
import json
import os
from datetime import datetime
import uuid
import subprocess
import time
from PIL import Image
import pytesseract

# ---------------- Tesseract Path ----------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---------------- Page Settings ----------------
st.set_page_config(page_title="ChatGPT + OCR", layout="wide")
CHAT_HISTORY_FILE = "chat_history.json"

# ---------------- Utility Functions ----------------
def now_iso():
    return datetime.now().isoformat(timespec="seconds")

def load_chats():
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE, "r") as f:
                chats = json.load(f)
        except Exception:
            return []
        changed = False
        for chat in chats:
            if "id" not in chat:
                chat["id"] = uuid.uuid4().hex
                changed = True
            if "name" not in chat:
                chat["name"] = derive_name_from_chat(chat)
                changed = True
        if changed:
            save_chats(chats)
        return chats
    return []

def save_chats(chats):
    with open(CHAT_HISTORY_FILE, "w") as f:
        json.dump(chats, f, indent=2)

def derive_name_from_chat(chat):
    if chat.get("name"):
        return chat["name"]
    for m in chat.get("messages", []):
        if m.get("role") == "user" and m.get("content"):
            return m["content"]
    return "Unnamed Chat"

def sort_key_desc(chat):
    ts = chat.get("timestamp", "")
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return datetime.fromtimestamp(0)

# ---------------- Ollama Backend ----------------
def query_gemma(prompt):
    try:
        result = subprocess.run(
            ["ollama", "run", "gemma:2b", prompt],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"⚠ Error: {e.stderr.strip()}"
    except Exception as e:
        return f"⚠ Unexpected Error: {str(e)}"

# ---------------- Load History ----------------
all_chats = load_chats()

# ---------------- Session State ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_id" not in st.session_state:
    st.session_state.chat_id = uuid.uuid4().hex
if "chat_name" not in st.session_state:
    st.session_state.chat_name = ""
if "ocr_text" not in st.session_state:
    st.session_state.ocr_text = ""

# ---------------- Sidebar ----------------
st.sidebar.title("Chat History")

if st.sidebar.button("+ New Chat"):
    # Reset all session state variables
    st.session_state.messages = []
    st.session_state.chat_id = uuid.uuid4().hex
    st.session_state.chat_name = ""
    st.session_state.ocr_text = ""

if st.sidebar.button("Clear Current Chat"):
    st.session_state.messages = []

if st.sidebar.button("Delete Current Chat"):
    all_chats = [c for c in all_chats if c.get("id") != st.session_state.chat_id]
    save_chats(all_chats)
    st.session_state.messages = []
    st.session_state.chat_id = uuid.uuid4().hex
    st.session_state.chat_name = ""
    st.session_state.ocr_text = ""

if st.sidebar.button("Delete All Chats"):
    if os.path.exists(CHAT_HISTORY_FILE):
        os.remove(CHAT_HISTORY_FILE)
    all_chats = []
    st.session_state.messages = []
    st.session_state.chat_id = uuid.uuid4().hex
    st.session_state.chat_name = ""
    st.session_state.ocr_text = ""

search_query = st.sidebar.text_input("🔎 Search chats by name")
st.sidebar.markdown("---")

sorted_chats = sorted(all_chats, key=sort_key_desc, reverse=True)
if not sorted_chats:
    st.sidebar.info("No saved chats yet. Start a new conversation!")

for chat in sorted_chats:
    chat_id = chat.get("id", uuid.uuid4().hex)
    chat["id"] = chat_id
    name = derive_name_from_chat(chat)

    if search_query and search_query.lower() not in name.lower():
        continue

    col1, col2 = st.sidebar.columns([10, 1])
    if col1.button(name, key=f"open_{chat_id}"):
        st.session_state.messages = chat.get("messages", [])
        st.session_state.chat_id = chat_id
        st.session_state.chat_name = name
        chat["timestamp"] = now_iso()
        save_chats(all_chats)
    if col2.button("🗑", key=f"del_{chat_id}"):
        all_chats = [c for c in all_chats if c.get("id") != chat_id]
        save_chats(all_chats)

# ---------------- Top Section: Title + Upload ----------------
st.title("New Conversation")
uploaded_files = st.file_uploader("Upload one or more images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

# ---------------- OCR Extraction + Automatic AI Response ----------------
if uploaded_files:
    st.session_state.ocr_text = ""
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        image_text = pytesseract.image_to_string(image).strip()
        if not image_text:
            continue
        st.session_state.ocr_text += f"{image_text}\n\n"

        # Display image in smaller width
        st.image(image, width=300)  # Adjust width as needed

        # Automatic AI response
        prompt = f"""You are an AI assistant.
Here is the text extracted from an image:

{image_text}

Provide a summary, insights, and key points from this text.
Answer questions that might arise from it."""

        st.session_state.messages.append({"role": "user", "content": "OCR text uploaded"})
        with st.chat_message("user"):
            st.markdown("OCR text uploaded")

        full_response = query_gemma(prompt)
        response_placeholder = st.empty()
        streaming_response = ""
        for char in full_response:
            streaming_response += char
            response_placeholder.markdown(streaming_response)
            time.sleep(0.005)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Save combined chat
    all_chats = [c for c in all_chats if c.get("id") != st.session_state.chat_id]
    all_chats.insert(0, {
        "id": st.session_state.chat_id,
        "name": "OCR Chat" if not st.session_state.chat_name else st.session_state.chat_name,
        "messages": st.session_state.messages,
        "timestamp": now_iso()
    })
    save_chats(all_chats)

# ---------------- Main Chat Area ----------------
for msg in st.session_state.messages[-100:]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- Chat Input for Follow-up Questions ----------------
if user_question := st.chat_input("Ask anything (AI will use OCR text if relevant)"):
    # Include OCR text for context
    if st.session_state.ocr_text.strip():
        final_prompt = f"""You are an AI assistant.
Here is the text extracted from uploaded images:

{st.session_state.ocr_text}

Answer the user's question using this text. Provide accurate answers, explanations, and insights.

User's question: {user_question}"""
    else:
        final_prompt = user_question

    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    if not st.session_state.chat_name:
        st.session_state.chat_name = user_question

    # Get AI response
    full_response = query_gemma(final_prompt)

    # Display streaming response
    response_placeholder = st.empty()
    streaming_response = ""
    for char in full_response:
        streaming_response += char
        response_placeholder.markdown(streaming_response)
        time.sleep(0.005)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Save chat
    all_chats = [c for c in all_chats if c.get("id") != st.session_state.chat_id]
    all_chats.insert(0, {
        "id": st.session_state.chat_id,
        "name": st.session_state.chat_name,
        "messages": st.session_state.messages,
        "timestamp": now_iso()
    })
    save_chats(all_chats)

import streamlit as st
from datetime import datetime
import requests, time, json
from io import StringIO
from PyPDF2 import PdfReader
from docx import Document
from PIL import Image
import pytesseract
import os
import platform
import subprocess
import base64
import re

def render_markdown(content):
    # Render code blocks as-is
    code_blocks = re.findall(r"```.*?```", content, flags=re.DOTALL)
    for i, block in enumerate(code_blocks):
        content = content.replace(block, f"@@BLOCK{i}@@")
    # Escape HTML outside code blocks
    content = html.escape(content)
    # Restore code blocks
    for i, block in enumerate(code_blocks):
        content = content.replace(f"@@BLOCK{i}@@", block)
    return content


# ===== TESSERACT CONFIG =====
def configure_tesseract():
    try:
        pytesseract.get_tesseract_version()
        return True
    except:
        pass
    if platform.system() == "Windows":
        paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
        for path in paths:
            if os.path.exists(path):
                try:
                    pytesseract.pytesseract.tesseract_cmd = path
                    subprocess.run([path,'--version'], timeout=10)
                    return True
                except:
                    continue
    return False

def optimize_image_for_ocr(image):
    try:
        if image.mode != "L":
            image = image.convert("L")
        if image.width > 1200:
            ratio = 1200 / image.width
            image = image.resize((1200, int(image.height*ratio)), Image.Resampling.LANCZOS)
    except:
        pass
    return image

def extract_text_with_ocr(uploaded_file):
    try:
        img = Image.open(uploaded_file)
        img = optimize_image_for_ocr(img)
        text = pytesseract.image_to_string(img, config='--oem 3 --psm 6')
        return text.strip(), True
    except Exception as e:
        return f"[OCR Error: {str(e)}]", False

def get_image_base64(uploaded_file):
    try:
        b64 = base64.b64encode(uploaded_file.getvalue()).decode()
        ext = uploaded_file.name.split('.')[-1].lower()
        mime = f"image/{ext}" if ext in ["png","jpg","jpeg"] else "image/jpeg"
        return f"data:{mime};base64,{b64}"
    except:
        return None

def derive_topic_from_text(text, max_words=4):
    """Get the 'topic' for a chat from first sentence and return up to max_words words."""
    if not text or not text.strip():
        return "New chat"
    # Normalize whitespace
    txt = re.sub(r'\s+', ' ', text.strip())
    # Split by sentence enders (., ?, !) or newline
    parts = re.split(r'[.?!\n]', txt)
    first = parts[0].strip() if parts else txt
    if not first:
        first = txt
    words = first.split()
    if len(words) <= max_words:
        topic = ' '.join(words)
    else:
        topic = ' '.join(words[:max_words]) + "..."
    # Capitalize first letter for neatness
    return topic[0].upper() + topic[1:] if topic else "New chat"

# ===== STREAMLIT CONFIG =====
st.set_page_config(
    page_title="ChatGPT Clone",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Main app styling */
    .stApp {
        background: #343541;
        color: #ececf1;
        font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: #202123 !important;
        min-width: 260px !important;
        max-width: 320px !important;
    }
    
    section[data-testid="stSidebar"] .stButton button {
        background: transparent;
        border: 1px solid #565869;
        color: white;
        border-radius: 6px;
        padding: 12px;
        width: 100%;
        text-align: left;
        font-size: 14px;
        margin: 2px 0;
    }
    
    section[data-testid="stSidebar"] .stButton button:hover {
        background: #2b2c2f;
    }
    
    /* Hide default elements */
    #MainMenu, header, footer {
        visibility: hidden;
    }
    
    /* Chat container */
    .chat-container {
        display: flex;
        flex-direction: column;
        height: calc(100vh - 120px);
        overflow-y: auto;
        padding: 20px 0;
        scroll-behavior: smooth;
    }
    
    /* Message styling */
    .message {
        display: flex;
        margin: 8px 0;
        width: 100%;
        padding: 0 12px;
    }
    
    .message.user {
        justify-content: flex-end;
        background: #343541;
    }
    
    .message.assistant {
        justify-content: flex-start;
        background: #343541;
    }
    
    .bubble {
        max-width: 70%;
        padding: 16px 20px;
        border-radius: 8px;
        line-height: 1.5;
        word-wrap: break-word;
        font-size: 16px;
    }
    
    .bubble.user {
        background: #000000;
        color: #ececf1;
        max-width: 50%;
    }
    
    .bubble.assistant {
        background: #000000;
        color: #ececf1;
        
    }
    
    /* File upload and image preview */
    .file-info-box {
        background: #2d2d2d;
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
        border-left: 4px solid #565869;
    }
    
    .image-preview {
        max-width: 100%;
        max-height: 300px;
        border-radius: 8px;
        margin: 8px 0;
    }
    
    .image-container {
        display: flex;
        justify-content: center;
        margin: 8px 0;
    }
    
    .extracted-text {
        background: #1a1a1a;
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
        border-left: 3px solid #565869;
        font-size: 14px;
        line-height: 1.4;
    }
    
    /* Floating input container */
    .stChatFloatingInput {
        position: fixed;
        bottom: 20px;
        left: 25%;
        right: 20px;
        background: #343541;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #565869;
        z-index: 1000;
        display: flex;
        align-items: center;
        gap: 10px;
        box-shadow: 0 -4px 20px rgba(0, 0, 0, 0.3);
    }
    
    .main .block-container {
        padding-bottom: 140px;
    }
    
    /* Chat input styling */
    .stChatInputContainer {
        background: #343541 !important;
        border: 1px solid #565869 !important;
        border-radius: 12px !important;
    }
    
    .stChatInputContainer textarea {
        background: #40414f !important;
        color: #ececf1 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px !important;
        font-size: 16px !important;
    }
    
    .stChatInputContainer textarea:focus {
        box-shadow: none !important;
        border: none !important;
    }
    
    .stChatInputContainer button {
        background: transparent !important;
        color: #ececf1 !important;
    }
    
    /* Chat history items */
    .chat-history-item {
        padding: 10px 12px;
        margin: 2px 0;
        border-radius: 6px;
        cursor: pointer;
        font-size: 14px;
        color: #ececf1;
        border: 1px solid transparent;
    }
    
    .chat-history-item:hover {
        background: #2b2c2f;
    }
    
    .chat-history-item.active {
        background: #343541;
        border-color: #565869;
    }
    
    /* Search box */
    .search-box {
        background: #202123;
        border: 1px solid #565869;
        border-radius: 6px;
        padding: 8px 12px;
        color: white;
        margin: 10px 0;
        width: 100%;
    }
    
    /* Scrollbar styling */
    .chat-container::-webkit-scrollbar {
        width: 6px;
    }
    
    .chat-container::-webkit-scrollbar-track {
        background: #2b2c2f;
        border-radius: 3px;
    }
    
    .chat-container::-webkit-scrollbar-thumb {
        background: #565869;
        border-radius: 3px;
    }
    
    .chat-container::-webkit-scrollbar-thumb:hover {
        background: #565869;
    }
    
    /* Welcome message */
    .welcome-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 70vh;
        text-align: center;
    }
    
    .welcome-title {
        font-size: 32px;
        font-weight: 600;
        margin-bottom: 20px;
        color: #ececf1;
    }
    
    .welcome-subtitle {
        font-size: 20px;
        color: #8e8ea0;
        margin-bottom: 40px;
    }
    
    /* Uploader styling */
    .uploader-container {
        position: absolute;
        bottom: 80px;
        left: 25%;
        right: 20px;
        background: #444654;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #565869;
        z-index: 999;
    }
    
    /* Hide file uploader text */
    div[data-testid="stFileUploader"] section {
        border: 1px solid #565869 !important;
        background: #444654 !important;
    }
    
    div[data-testid="stFileUploader"] > div > div {
        display: none !important;
    }
    
    /* Three-dot menu styling */
    .menu-dots {
        background: transparent;
        border: none;
        color: #8e8ea0;
        font-size: 18px;
        cursor: pointer;
        padding: 4px 8px;
        border-radius: 4px;
    }
    
    .menu-dots:hover {
        background: #2b2c2f;
        color: #ececf1;
    }
</style>
""", unsafe_allow_html=True)

# ===== SESSION STATE =====
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}
if "active_chat" not in st.session_state:
    st.session_state.active_chat = None
if "tesseract_configured" not in st.session_state:
    st.session_state.tesseract_configured = configure_tesseract()
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None
if "show_uploader" not in st.session_state:
    st.session_state.show_uploader = False
if "search_query" not in st.session_state:
    st.session_state.search_query = ""
if "menu_open" not in st.session_state:
    st.session_state.menu_open = None
if "rename_buffers" not in st.session_state:
    st.session_state.rename_buffers = {}

# ===== NEW CHAT =====
def new_chat(name=None):
    chat_id = str(datetime.now().timestamp())
    chat_name = name or "New chat"
    st.session_state.chat_history[chat_id] = {
        "name": chat_name,
        "messages": [],
        "created": datetime.now(),
        "latest_file": None
    }
    st.session_state.active_chat = chat_id
    st.session_state.show_uploader = False

# ===== FILE EXTRACTION =====
def extract_text_from_file(uploaded_file):
    text = ""
    file_type = uploaded_file.type
    file_info = {
        "name": uploaded_file.name,
        "type": file_type,
        "size": len(uploaded_file.getvalue())
    }
    
    try:
        if file_type == "application/pdf":
            reader = PdfReader(uploaded_file)
            for page in reader.pages:
                text += page.extract_text() or ""
            file_info["pages"] = len(reader.pages)
            file_info["extracted"] = bool(text.strip())
            
        elif file_type == "text/plain":
            text = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
            file_info["extracted"] = bool(text.strip())
            
        elif file_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
            doc = Document(uploaded_file)
            for p in doc.paragraphs:
                text += p.text + "\n"
            file_info["extracted"] = bool(text.strip())
            
        elif file_type in ["image/png", "image/jpeg", "image/jpg"]:
            if st.session_state.tesseract_configured:
                text, success = extract_text_with_ocr(uploaded_file)
                file_info["extracted"] = success
                file_info["ocr_used"] = True
            else:
                text = f"[Image file: {uploaded_file.name}]"
                file_info["extracted"] = False
        else:
            text = f"[Unsupported file type: {file_type}]"
            file_info["extracted"] = False
            
    except Exception as e:
        text = f"[Error reading file: {str(e)}]"
        file_info["extracted"] = False
        file_info["error"] = str(e)
        
    return text.strip(), file_info

# ===== QUERY AI FUNCTION =====
def query_ollama_stream(prompt, context=None):
    url = "http://localhost:11434/api/chat"
    messages = []
    
    if context and context.strip() and not context.startswith("["):
        messages.append({
            "role": "system", 
            "content": f"User uploaded file content:\n{context[:4000]}\nAnswer based on this content when relevant."
        })
    
    messages.append({"role": "user", "content": prompt})
    
    data = {
        "model": "llama2",
        "messages": messages,
        "stream": True
    }
    
    try:
        with requests.post(url, json=data, stream=True, timeout=60) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    text = line.decode("utf-8")
                    if text.strip() == "data: [DONE]": 
                        break
                    try:
                        chunk = json.loads(text.replace("data:", ""))
                        if "message" in chunk and "content" in chunk["message"]:
                            yield chunk["message"]["content"]
                    except:
                        continue
    except Exception as e:
        yield f"[Error connecting to Ollama: {str(e)}]"

# ===== SIDEBAR =====
with st.sidebar:
    st.markdown('<h3 style="color:white; text-align: center; margin-bottom: 20px;">🤖 ChatGPT Clone</h3>', unsafe_allow_html=True)
    
    # New Chat Button
    if st.button("+ New chat", use_container_width=True, key="new_chat_btn"):
        new_chat()
        st.rerun()
    
    st.markdown("---")
    
    # Search Chats
    st.markdown("**Search chats**")
    search_query = st.text_input(
        "Search chat titles", 
        value=st.session_state.search_query,
        key="search_input",
        label_visibility="collapsed"
    )
    st.session_state.search_query = search_query
    
    st.markdown("**Chat History**")
    
    # Chat History
    if not st.session_state.chat_history:
        st.info("No chats yet. Start a conversation!")
    else:
        # Filter chats based on search
        filtered_chats = {}
        for chat_id, chat in st.session_state.chat_history.items():
            if not search_query or search_query.lower() in chat["name"].lower():
                filtered_chats[chat_id] = chat
        
        if not filtered_chats:
            st.info("No chats match your search.")
        else:
            # Display chats (newest first)
            for chat_id, chat in sorted(
                filtered_chats.items(), 
                key=lambda x: x[1]["created"], 
                reverse=True
            ):
                is_active = st.session_state.active_chat == chat_id
                active_class = "active" if is_active else ""
                
                cols = st.columns([0.8, 0.2])
                
                with cols[0]:
                    if st.button(
                        chat["name"], 
                        key=f"chat_{chat_id}",
                        use_container_width=True
                    ):
                        st.session_state.active_chat = chat_id
                        st.session_state.menu_open = None
                        st.rerun()
                
                with cols[1]:
                    if st.button("⋯", key=f"menu_{chat_id}", help="Chat options"):
                        if st.session_state.menu_open == chat_id:
                            st.session_state.menu_open = None
                        else:
                            st.session_state.menu_open = chat_id
                        st.rerun()
                
                # Chat menu options
                if st.session_state.menu_open == chat_id:
                    menu_cols = st.columns(2)
                    
                    with menu_cols[0]:
                        if st.button("Rename", key=f"rename_{chat_id}"):
                            if chat_id not in st.session_state.rename_buffers:
                                st.session_state.rename_buffers[chat_id] = chat["name"]
                            st.session_state.menu_open = f"rename_{chat_id}"
                            st.rerun()
                    
                    with menu_cols[1]:
                        if st.button("Delete", key=f"delete_{chat_id}"):
                            if st.session_state.active_chat == chat_id:
                                remaining = list(st.session_state.chat_history.keys())
                                remaining.remove(chat_id)
                                st.session_state.active_chat = remaining[0] if remaining else None
                            del st.session_state.chat_history[chat_id]
                            st.session_state.menu_open = None
                            st.rerun()
                
                # Rename input
                if st.session_state.menu_open == f"rename_{chat_id}":
                    if chat_id not in st.session_state.rename_buffers:
                        st.session_state.rename_buffers[chat_id] = chat["name"]
                    
                    new_name = st.text_input(
                        "New name",
                        value=st.session_state.rename_buffers[chat_id],
                        key=f"rename_input_{chat_id}"
                    )
                    
                    name_cols = st.columns(2)
                    with name_cols[0]:
                        if st.button("Save", key=f"save_rename_{chat_id}"):
                            if new_name.strip():
                                st.session_state.chat_history[chat_id]["name"] = new_name.strip()
                            st.session_state.menu_open = None
                            st.rerun()
                    with name_cols[1]:
                        if st.button("Cancel", key=f"cancel_rename_{chat_id}"):
                            st.session_state.menu_open = None
                            st.rerun()

# ===== MAIN CHAT AREA =====
main_col = st.columns([1])[0]

with main_col:
    if st.session_state.active_chat is None:
        # Welcome screen
        st.markdown("""
        <div class="welcome-container">
            <div class="welcome-title">What can I help you with today?</div>
            <div class="welcome-subtitle">Ask anything (you can refer to uploaded images or documents)</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        chat = st.session_state.chat_history[st.session_state.active_chat]
        chat_container = st.container()
        
        with chat_container:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            
            if not chat["messages"]:
                st.markdown("""
                <div style="text-align: center; padding: 40px; color: #8e8ea0;">
                    <h3>Start a conversation</h3>
                    <p>Send a message or upload a file to get started</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                for msg_entry in chat["messages"]:
                    role = msg_entry["role"]
                    msg = msg_entry["message"]
                    css_class = "user" if role == "user" else "assistant"
                    
                    # Handle file attachments
                    attachment_html = ""
                    if role == "user" and msg_entry.get("uploaded_files"):
                        for fname, finfo in msg_entry["uploaded_files"].items():
                            if fname in msg_entry.get("uploaded_images", {}):
                                img_data = msg_entry["uploaded_images"][fname]
                                attachment_html += f'''
                                <div class="file-info-box">
                                    📎 {fname}
                                    <div class="image-container">
                                        <img src="{img_data['base64']}" class="image-preview">
                                    </div>
                                </div>
                                '''
                            else:
                                attachment_html += f'<div class="file-info-box">📎 {fname}</div>'
                    
                    if attachment_html:
                        full_msg = f'{attachment_html}<br>{msg}'
                    else:
                        full_msg = msg
                    
                    st.markdown(
                        f'<div class="message {css_class}"><div class="bubble {css_class}">{full_msg}</div></div>', 
                        unsafe_allow_html=True
                    )
            
            st.markdown('</div>', unsafe_allow_html=True)

# ===== FLOATING INPUT AREA =====
with st.container():
    st.markdown('<div class="stChatFloatingInput">', unsafe_allow_html=True)
    
    cols = st.columns([1, 8, 1])
    
    with cols[0]:
        if st.button("📎", key="open_ocr", help="Upload image for OCR", use_container_width=True):
            st.session_state.show_uploader = not st.session_state.show_uploader
            st.rerun()
    
    with cols[1]:
        prompt = st.chat_input("Ask anything (you can refer to uploaded image)...")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ===== FILE UPLOADER =====
if st.session_state.show_uploader:
    with st.container():
        st.markdown('<div class="uploader-container">', unsafe_allow_html=True)
        st.write("Upload files for OCR or text extraction")
        
        uploaded_file = st.file_uploader(
            "Choose files",
            type=["pdf", "docx", "txt", "png", "jpg", "jpeg"],
            label_visibility="collapsed",
            key="floating_uploader"
        )
        
        if uploaded_file and uploaded_file != st.session_state.last_uploaded_file:
            st.session_state.last_uploaded_file = uploaded_file
            
            if st.session_state.active_chat is None:
                new_chat()
            
            chat = st.session_state.chat_history[st.session_state.active_chat]
            
            # Process uploaded file
            extracted_text, file_info = extract_text_from_file(uploaded_file)
            uploaded_images = {}
            
            if file_info.get("ocr_used", False):
                b64 = get_image_base64(uploaded_file)
                if b64:
                    uploaded_images[uploaded_file.name] = {
                        "base64": b64, 
                        "text": extracted_text or "No text extracted"
                    }
            
            # Save as latest file for future input
            chat["latest_file"] = {
                "name": uploaded_file.name,
                "file_info": file_info,
                "uploaded_images": uploaded_images,
                "text": extracted_text
            }
            
            st.success(f"File '{uploaded_file.name}' uploaded successfully!")
            st.session_state.show_uploader = False
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

# ===== HANDLE USER MESSAGE =====
if prompt:
    if st.session_state.active_chat is None:
        new_chat()
    
    chat = st.session_state.chat_history[st.session_state.active_chat]
    
    # Capture file context
    file_snapshot = {}
    uploaded_images_snapshot = {}
    file_context = None
    
    if chat.get("latest_file"):
        lf = chat["latest_file"]
        file_snapshot[lf["name"]] = lf["file_info"]
        uploaded_images_snapshot = lf.get("uploaded_images", {})
        file_context = lf["text"]
    
    # If this is the FIRST message in the chat, set the chat 'topic' as name
    if len(chat["messages"]) == 0:
        topic = derive_topic_from_text(prompt, max_words=4)
        chat["name"] = topic
    
    # Append user message
    chat["messages"].append({
        "role": "user",
        "message": prompt,
        "uploaded_files": file_snapshot,
        "uploaded_images": uploaded_images_snapshot,
        "file_context": file_context
    })
    
    st.rerun()
# ===== STREAM AI RESPONSE =====
if st.session_state.active_chat:
    chat = st.session_state.chat_history[st.session_state.active_chat]
    
    if chat["messages"] and chat["messages"][-1]["role"] == "user":
        last_msg = chat["messages"][-1]
        user_prompt = last_msg["message"]
        file_context = last_msg.get("file_context")
        
        with chat_container:
            placeholder = st.empty()
            buffer = ""
            bot_reply = ""
            last_update = time.time()
            
            import html  # FIX: Added to escape HTML
            
            # Stream the AI response
            for token in query_ollama_stream(user_prompt, file_context):
                buffer += token
                current_time = time.time()
                
                # Update display periodically for smoother streaming
                if current_time - last_update > 0.1 or len(buffer) > 50:
                    bot_reply += buffer
                    safe_bot_reply = render_markdown(bot_reply) # FIX: Escape HTML
                    placeholder.markdown(
                        f'<div class="message assistant"><div class="bubble assistant">{safe_bot_reply}▌</div></div>', 
                        unsafe_allow_html=True
                    )
                    buffer = ""
                    last_update = current_time
            
            # Add remaining buffer
            bot_reply += buffer
            
            # Final display without cursor
            safe_bot_reply = bot_reply  # FIX: Escape HTML
            placeholder.markdown(
                f'<div class="message assistant"><div class="bubble assistant">{safe_bot_reply}</div></div>', 
                unsafe_allow_html=True
            )
            
            # Save assistant message
            chat["messages"].append({
                "role": "assistant",
                "message": bot_reply,
                "uploaded_files": last_msg.get("uploaded_files"),
                "uploaded_images": last_msg.get("uploaded_images"),
                "file_context": last_msg.get("file_context")
            })
            
            # Clear latest file after use
            chat["latest_file"] = None

import streamlit as st
import uuid, datetime, json, os, subprocess, socket
from PIL import Image, ImageOps, ImageFilter
import io
import pandas as pd
import time

# --- Dependency Imports ---
# Ensure these are installed: pip install pytesseract pdfplumber python-docx ollama pandas Pillow
try:
    import pytesseract
    # CRITICAL: Windows users may need to set Tesseract path:
    # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
except Exception:
    pytesseract = None

try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    from docx import Document
except Exception:
    Document = None

try:
    import ollama
    OLLAMA_AVAILABLE = True
except Exception:
    ollama = None
    OLLAMA_AVAILABLE = False

# Page config
st.set_page_config(page_title="CodeGenesis Explainer", layout="wide")

# --- Paths / storage ---
DATA_FILE = "chats.json"
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ----------------------------------------------------------------------------------
# SYSTEM INSTRUCTION
# ----------------------------------------------------------------------------------
SYSTEM_INSTRUCTION = (
    "You are CodeGenesis, an expert AI assistant specialized in generating, explaining, and correcting code. "
    "You are also a highly knowledgeable and comprehensive general assistant, like a top-tier LLM (e.g., ChatGPT or Gemini). "
    "Your response should directly and completely address the user's request, whether it's a general question or a file/code analysis request. "
    "For Code Analysis, Fixing, or Generation requests, provide the necessary explanation and corrections, and always include the complete and correct code block using proper language syntax highlighting (e.e.g., python...). "
    "For General Questions, provide a comprehensive, accurate, and detailed answer directly. Your tone must be professional, helpful, and friendly."
)

# ---------- Helper Functions ----------
def now_iso(): return datetime.datetime.now().isoformat()
def make_chat(title=None):
    return {"id": uuid.uuid4().hex, "title": title or "New Chat", "messages": [], "created_at": now_iso()}
def load_chats():
    if os.path.exists(DATA_FILE):
        try: return json.load(open(DATA_FILE, "r", encoding="utf-8"))
        except Exception: return []
    return []
def save_chats(chats):
    open(DATA_FILE, "w", encoding="utf-8").write(json.dumps(chats, indent=2, ensure_ascii=False))
def ensure_ollama_running():
    if not OLLAMA_AVAILABLE: return False
    try:
        with socket.create_connection(("127.0.0.1", 11434), timeout=0.2): return True
    except Exception: return False
def get_installed_models():
    if not OLLAMA_AVAILABLE or not ensure_ollama_running(): return ["(Ollama not available)"]
    try:
        res = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=False, timeout=5)
        lines = [l.strip().split()[0] for l in res.stdout.splitlines()[1:] if l.strip()]
        return lines or ["gemma:2b"]
    except Exception:
        return ["(Error listing models)"]
def preprocess_image_for_ocr(pil_img):
    try:
        img = pil_img.convert("L")
        w, h = img.size
        if img.getpixel((0,0)) + img.getpixel((w-1, h-1)) < 50: img = ImageOps.invert(img)
        img = ImageOps.autocontrast(img)
        if max(w, h) < 1000:
            scale = max(1, int(1200 / max(w, h)))
            img = img.resize((w * scale, h * scale), Image.Resampling.LANCZOS)
        img = img.filter(ImageFilter.SHARPEN)
        img = img.point(lambda p: 255 if p > 150 else 0)
        return img
    except Exception:
        return pil_img
def extract_text(path):
    ext = path.split(".")[-1].lower()
    try:
        if ext in ["jpg", "jpeg", "png", "bmp", "tiff"]:
            if not pytesseract: return "(ERROR: pytesseract not installed.)"
            try:
                with Image.open(path) as im:
                    text = pytesseract.image_to_string(im, config='--psm 3')
                    if not text.strip() or len(text.strip().split()) < 10:
                        im_processed = preprocess_image_for_ocr(im)
                        text = pytesseract.image_to_string(im_processed, config='--psm 6')
                    return text.strip() or "(No significant text detected in image.)"
            except pytesseract.TesseractNotFoundError: return "(ERROR: Tesseract executable not found.)"
            except Exception as e: return f"(OCR failed: {type(e).name}: {e}.)"
        if ext == "pdf":
            # PDF support is here
            if not pdfplumber: return "(ERROR: pdfplumber not installed.)"
            txt = ""
            with pdfplumber.open(path) as pdf:
                for p in pdf.pages: txt += (p.extract_text() or "") + "\n"
            return txt.strip() or "(No text detected in PDF.)"
        if ext == "docx":
            # DOCX support is here
            if not Document: return "(ERROR: python-docx not installed.)"
            doc = Document(path)
            txt = "\n".join(p.text for p in doc.paragraphs)
            return txt.strip() or "(Empty DOCX file.)"
        if ext == "txt":
            with open(path, "r", encoding="utf-8", errors="ignore") as f: return f.read()
        if ext == "csv": return pd.read_csv(path, nrows=200).to_string()
        return "(Unsupported file type for extraction.)"
    except Exception as e: return f"(Extraction error: {type(e).name}: {e})"

# ---------- Session State Initialization ----------
if "chats" not in st.session_state: st.session_state["chats"] = load_chats()
if not st.session_state["chats"]:
    new_chat = make_chat("New Chat")
    st.session_state["chats"].append(new_chat)
    save_chats(st.session_state["chats"])
elif st.session_state["chats"][0]["title"] == "Welcome":
    st.session_state["chats"][0]["title"] = "New Chat"
    st.session_state["chats"][0]["messages"] = []
    save_chats(st.session_state["chats"])
if "current_chat" not in st.session_state or st.session_state["current_chat"] not in [c["id"] for c in st.session_state["chats"]]: st.session_state["current_chat"] = st.session_state["chats"][0]["id"]
if "show_upload" not in st.session_state: st.session_state["show_upload"] = False
if "llm_running" not in st.session_state: st.session_state["llm_running"] = False
if "new_message_to_process" not in st.session_state: st.session_state["new_message_to_process"] = False
if "message_content" not in st.session_state: st.session_state["message_content"] = ""
if "sidebar_search" not in st.session_state: st.session_state["sidebar_search"] = ""
if "staged_file_context" not in st.session_state: st.session_state["staged_file_context"] = None 

chats = st.session_state["chats"]
chat = next((c for c in chats if c["id"] == st.session_state["current_chat"]), None)
if chat is None:
    chat = make_chat()
    st.session_state["chats"].insert(0, chat)
    st.session_state["current_chat"] = chat["id"]
    save_chats(st.session_state["chats"])

# ---------- Core Logic Functions ----------
def build_file_prompt(file_info, action_text):
    """Generates the comprehensive LLM prompt for file analysis."""
    extracted_content_chunk = file_info['extracted_text'][:12000]
    
    extraction_status = "Extraction successful."
    if file_info['extracted_text'].startswith("(ERROR"):
        extraction_status = f"Extraction failed with error: {file_info['extracted_text']}. Warn the user."
    elif "No significant text detected" in file_info['extracted_text']:
        extraction_status = "Extraction warning: Little to no relevant text/code was detected. Advise the user to verify the content."
    
    return f"""
--- FILE ANALYSIS CONTEXT ---
File Name: {file_info['name']}
File Type: {file_info['type']}
Extraction Status: {extraction_status}
CRITICAL INSTRUCTION: This is a combined file and action request. Process the user's action against the content below.

--- EXTRACTED CONTENT (12000 characters max) ---
{extracted_content_chunk}

--- USER ACTION ---
User's requested action: {action_text}
"""

def handle_send_click():
    """
    Handles text message submission. Combines staged file content (if any) with the action text.
    Modified to separate file info/path from user text for display order.
    """
    msg_key = st.session_state.get("message_content_input", "").strip()
    
    if msg_key:
        if chat["title"] == "New Chat":
            chat["title"] = msg_key[:30] + ("..." if len(msg_key) > 30 else "")
        
        # Check for staged file
        if st.session_state["staged_file_context"]:
            file_info = st.session_state["staged_file_context"]
            
            # 1. Prepare visible message structure
            user_action_content = msg_key
            file_icon = "📷" if file_info.get("type").startswith("image") else "📎"
            
            # The file summary is now a separate element, not part of 'content'
            file_summary_content = f"{file_icon} File {file_info['name']} uploaded."

            # Create the message object
            msg_obj = {
                "role": "user", 
                "content": user_action_content, # This is the user's question, displayed LAST
                "file_summary": file_summary_content # Displayed FIRST
            }

            if file_info.get("type").startswith("image"):
                msg_obj["image_path"] = file_info['path'] # Image displayed in the middle
            
            chat["messages"].append(msg_obj)
            
            # 2. Build the hidden combined prompt for the LLM
            llm_prompt = build_file_prompt(file_info, msg_key)
            st.session_state["staged_file_context"] = None # Clear the staged file context
            
        else:
            # Regular chat submission
            chat["messages"].append({"role": "user", "content": msg_key})
            llm_prompt = f"USER QUERY (GENERAL/CONVERSATIONAL/NEW CODE REQUEST): {msg_key}\n\n[End of query. Provide a direct and complete answer, using code blocks if requested.]"

        # 3. Add hidden prompt for LLM
        chat["messages"].append({"role": "user", "content": llm_prompt, "hidden": True})
        
        st.session_state.message_content_input = ""
        st.session_state.new_message_to_process = True
        save_chats(st.session_state["chats"])

def handle_file_upload_only():
    """
    Processes the uploaded file, extracts text, and stages it in session state.
    """
    uploaded_file = st.session_state.get("file_upload_widget")
    
    if uploaded_file is None or uploaded_file.size == 0: 
        st.session_state["file_upload_widget"] = None
        return
        
    st.session_state["show_upload"] = False # Close the upload expander

    filename = f"{uuid.uuid4().hex}_{uploaded_file.name}"
    path = os.path.join(UPLOAD_DIR, filename)
    
    with st.spinner(f"Processing '{uploaded_file.name}' (OCR / extract)..."):
        with open(path, "wb") as f:
            f.write(uploaded_file.getvalue())
        extracted = extract_text(path)
        
    if chat["title"] == "New Chat":
        chat["title"] = uploaded_file.name.replace(".", "_")[:30]

    # Stage the file context.
    st.session_state["staged_file_context"] = {
        "name": uploaded_file.name,
        "type": uploaded_file.type,
        "path": path,
        "extracted_text": extracted
    }
    
    st.session_state["file_upload_widget"] = None 
    # st.rerun() removed here to prevent the "no-op" error.

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("CodeGenesis Explainer")
    st.markdown("---")
    
    if OLLAMA_AVAILABLE:
        if not ensure_ollama_running():
            st.error("❌ Ollama server not reachable. Run ollama serve in your terminal.")
    else:
        st.info("ℹ Ollama library not installed. Install with pip install ollama.")

    st.subheader("Model Selection")
    models = get_installed_models()
    default_model_index = 0
    is_model_selection_disabled = False
    
    preferred_models = ["llama3:8b", "codellama:7b", "gemma:2b"]
    for i, model_name in enumerate(preferred_models):
        if model_name in models:
            try:
                default_model_index = models.index(model_name)
                break
            except ValueError:
                pass
    
    if not models or "(Ollama not available)" in models[0]:
        is_model_selection_disabled = True
    
    model = st.selectbox("Model", models, index=default_model_index, key="selected_model", disabled=is_model_selection_disabled)
    st.markdown("---")
    
    if st.button("➕ New Chat", use_container_width=True):
        nc = make_chat()
        st.session_state["chats"].insert(0, nc)
        st.session_state["current_chat"] = nc["id"]
        save_chats(st.session_state["chats"])
        st.rerun()

    st.subheader("Chat History")
    st.text_input("🔍 Search", key="sidebar_search", placeholder="Search chat titles...")
    
    search_term = st.session_state.sidebar_search.lower()
    chats_list_filtered = [c for c in st.session_state["chats"] if search_term in c["title"].lower()] if search_term else st.session_state["chats"]
    
    for c in chats_list_filtered:
        cols = st.columns([0.8, 0.2])
        with cols[0]:
            button_style = "primary" if c["id"] == st.session_state["current_chat"] else "secondary"
            if st.button(c["title"], key=f"chat-{c['id']}", use_container_width=True, type=button_style):
                st.session_state["current_chat"] = c["id"]
                st.rerun()
        with cols[1]:
            if st.button("🗑", key=f"del-{c['id']}", use_container_width=True, help="Delete chat", type="secondary"):
                st.session_state["chats"] = [x for x in st.session_state["chats"] if x["id"] != c["id"]]
                save_chats(st.session_state["chats"])
                if not st.session_state["chats"]:
                    new_chat_after_delete = make_chat("New Chat")
                    st.session_state["chats"].append(new_chat_after_delete)
                    st.session_state["current_chat"] = new_chat_after_delete["id"]
                elif st.session_state["current_chat"] == c["id"]:
                    st.session_state["current_chat"] = st.session_state["chats"][0]["id"]
                st.rerun()


# ---------------- MAIN CHAT DISPLAY (Image size fixed to 300px) ----------------
chat_container = st.container()
with chat_container:
    for msg in chat["messages"]:
        if msg.get("hidden"):
            continue
        with st.chat_message(msg["role"]):
            
            # 1. Display File Summary FIRST
            if msg.get("file_summary"):
                st.markdown(msg["file_summary"])

            # 2. Display Image SECOND (Fixed small width)
            if msg.get("image_path"):
                try:
                    # Image size fixed to 300px width for smaller display
                    st.image(msg["image_path"], width=300) 
                except Exception:
                    st.write("(Could not display image)")

            # 3. Display User's Question/Content LAST
            st.markdown(msg["content"])
            
    # Custom CSS for fixed input bar and styling
    st.markdown("""
    <style>
    /* Hide the Streamlit header */
    .stApp > header { display: none; }
    /* Add padding to the main content so it doesn't overlap the fixed input bar */
    .main { padding-bottom: 160px; }
    
    /* Style for the fixed input container */
    .fixed-input-container {
        position: fixed;
        bottom: 0;
        z-index: 1000;
        background: #f0f2f6; /* Match Streamlit's background color */
        border-top: 1px solid #ccc;
        padding: 12px 12px 12px 18px;
        left: 0;
        width: 100%;
        box-sizing: border-box;
    }

    /* Adjust the fixed input container width for non-mobile views (with sidebar) */
    @media (min-width: 768px) {
        .fixed-input-container {
            left: 300px; /* Assumes sidebar is 300px wide */
            width: calc(100% - 300px);
        }
    }
    
    /* General sidebar button styling for aesthetics */
    section[data-testid="stSidebar"] button[data-testid="baseButton-primary"] {
        background-color: #FFFFFF !important;
        border-color: #DDDDDD !important;
        color: #1A1A1A !important;
        box-shadow: 0 0 2px 0 rgba(0,0,0,0.1);
    }
    
    section[data-testid="stSidebar"] button[data-testid="baseButton-secondary"] {
        background-color: transparent !important;
        border-color: transparent !important;
        color: #333333 !important;
    }
    
    </style>
    """, unsafe_allow_html=True)


# ---------------- LLM GENERATION LOGIC ----------------
if st.session_state["new_message_to_process"] and not st.session_state["llm_running"]:
    st.session_state["llm_running"] = True
    st.session_state["new_message_to_process"] = False

    # Scroll to bottom before starting generation
    st.markdown('<script>window.scrollTo(0, document.body.scrollHeight);</script>', unsafe_allow_html=True)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        partial_response_content = ""

        final_prompt = next((m["content"] for m in reversed(chat["messages"]) if m.get("hidden") and m["role"] == "user"), None)
        if final_prompt is None:
            final_prompt = chat["messages"][-1]["content"] if chat["messages"] else "Please provide a valid query."
        
        if not OLLAMA_AVAILABLE or not ensure_ollama_running() or is_model_selection_disabled:
            err_msg = (
                "⚠ Local LLM (Ollama) not available or server is down. "
                "Action: Please ensure you have Ollama installed, run ollama serve in your terminal, and pull the desired model."
            )
            placeholder.markdown(err_msg)
            chat["messages"].append({"role": "assistant", "content": err_msg})
            save_chats(chats)
        else:
            messages_for_ollama = [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": final_prompt} 
            ]

            try:
                model_to_use = model if model not in ["(Ollama not available)"] else "gemma:2b"
                
                for chunk in ollama.chat(model=model_to_use, messages=messages_for_ollama, stream=True):
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        partial_response_content += content
                        placeholder.markdown(partial_response_content + "▌")
                
                placeholder.markdown(partial_response_content)

                chat["messages"].append({"role": "assistant", "content": partial_response_content})
                save_chats(chats)

            except Exception as e:
                err_msg = f"❌ Error during LLM generation: {type(e).name}: {e}. Ensure model {model} is pulled."
                placeholder.markdown(err_msg)
                chat["messages"].append({"role": "assistant", "content": err_msg})
                save_chats(chats)

    st.session_state["llm_running"] = False
    st.rerun() 

# ---------------- FIXED INPUT BAR / UPLOAD ----------------
st.markdown('<div class="fixed-input-container">', unsafe_allow_html=True)

if st.session_state["show_upload"]:
    with st.expander("Upload code/document for analysis (Image, PDF, DOCX, TXT, CSV)", expanded=True):
        st.file_uploader(
            "Upload file (max size is determined by Streamlit settings)",
            type=["jpg", "jpeg", "png", "pdf", "docx", "txt", "csv"], # <--- Includes PDF and DOCX
            key="file_upload_widget", 
            label_visibility="collapsed",
            disabled=st.session_state["llm_running"],
            on_change=handle_file_upload_only
        )

col_plus, col_text = st.columns([0.07, 0.93], gap="small")
with col_plus:
    if st.button("➕", key="upload_toggle_btn", disabled=st.session_state["llm_running"], help="Toggle file upload panel", use_container_width=True):
        st.session_state["show_upload"] = not st.session_state["show_upload"]

with col_text:
    placeholder_text = "Ask any question...."
    if st.session_state["staged_file_context"]:
        # CRITICAL: This guides the user when a file is ready.
        placeholder_text = f"File '{st.session_state['staged_file_context']['name']}' is ready. Enter your question (e.g., 'Fix the code', 'Summarize this')."
        
    st.text_input(
        "Type message here...",
        key="message_content_input",
        label_visibility="collapsed",
        disabled=st.session_state["llm_running"],
        on_change=handle_send_click, # This triggers the full LLM generation
        placeholder=placeholder_text
    )

st.markdown('</div>', unsafe_allow_html=True)

import streamlit as st
import os
import json
import shutil
import tempfile
from model import query_ollama
from input_preprocessing import process_input
from working_ocr_model import OptimizedOCRModel
import asyncio

if "ocr_model" not in st.session_state:
    st.session_state.ocr_model = OptimizedOCRModel()


# ---- Base Path ----
BASE_PATH = "/Users/dilipdilip/Downloads/TalenticaAssignment/session_data"
os.makedirs(BASE_PATH, exist_ok=True)

# ---- Load Existing Sessions ----
def load_existing_sessions():
    sessions = {}
    for session_name in os.listdir(BASE_PATH):
        session_path = os.path.join(BASE_PATH, session_name)
        if os.path.isdir(session_path):
            # Load chat
            chat_path = os.path.join(session_path, "chat.json")
            if os.path.exists(chat_path):
                try:
                    with open(chat_path, "r") as f:
                        chat = json.load(f)
                except json.JSONDecodeError:
                    chat = []
            else:
                chat = []

            # OCR data
            ocr_data = []
            ocr_folder = os.path.join(session_path, "ocr")
            if os.path.exists(ocr_folder):
                for file in os.listdir(ocr_folder):
                    with open(os.path.join(ocr_folder, file), "r") as f:
                        ocr_data.append({"filename": file.replace(".txt", ""), "text": f.read()})

            # Images (just references, not copying back)
            images = []
            img_folder = os.path.join(session_path, "images")
            if os.path.exists(img_folder):
                for file in os.listdir(img_folder):
                    images.append({"filename": file, "path": os.path.join(img_folder, file)})

            sessions[session_name] = {"chat": chat, "images": images, "ocr": ocr_data}
    return sessions

# ---- Initialize Session Data ----
if "sessions" not in st.session_state:
    st.session_state.sessions = load_existing_sessions()

if "current_session" not in st.session_state:
    if st.session_state.sessions:
        # Pick first session if exists
        st.session_state.current_session = list(st.session_state.sessions.keys())[0]
    else:
        session_name = "Session 1"
        st.session_state.sessions[session_name] = {"chat": [], "images": [], "ocr": []}
        st.session_state.current_session = session_name

# Initialize additional session state variables
if "bookena" not in st.session_state:
    st.session_state.bookena = False
if "parsed_bill" not in st.session_state:
    st.session_state.parsed_bill = False



def create_new_session():
    new_name = f"Session {len(st.session_state.sessions) + 1}"
    st.session_state.sessions[new_name] = {"chat": [], "images": [], "ocr": []}
    st.session_state.current_session = new_name
    st.session_state.bookena = False
    st.session_state.parsed_bill = False

def save_session_data(session_name, session_data):
    session_path = os.path.join(BASE_PATH, session_name)
    os.makedirs(session_path, exist_ok=True)

    # --- Save chat ---
    with open(os.path.join(session_path, "chat.json"), "w") as f:
        json.dump(session_data["chat"], f, indent=4)

    # --- Save images ---
    images_path = os.path.join(session_path, "images")
    os.makedirs(images_path, exist_ok=True)
    for img in session_data["images"]:
        target_path = os.path.join(images_path, img["filename"])
        if not os.path.exists(target_path):
            shutil.copy(img["path"], target_path)

    # --- Save OCR text ---
    ocr_path = os.path.join(session_path, "ocr")
    os.makedirs(ocr_path, exist_ok=True)
    for o in session_data["ocr"]:
        ocr_file = os.path.join(ocr_path, f"{o['filename']}.txt")
        with open(ocr_file, "w") as f:
            f.write(o["text"])

# ---- Sidebar ----
st.sidebar.title("Sessions")
for s in st.session_state.sessions.keys():
    if st.sidebar.button(s):
        st.session_state.current_session = s
if st.sidebar.button("➕ New Session"):
    create_new_session()

current_session = st.session_state.current_session
session_data = st.session_state.sessions[current_session]

# ---- Display Chat (only text) ----
st.title(f"Chat - {current_session}")
for msg in session_data["chat"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["text"])


# ---- Chat Input ----
user_input = st.chat_input("Type your message - inlcude export for json export, invoice or parse for bill parsing ...")
uploaded_image = st.file_uploader("Attach Image (optional)", type=["png", "jpg", "jpeg"], key="img_upload")

# uploaded_image = None




if st.session_state.bookena:
    json_download_path = os.path.join(BASE_PATH, current_session, "json_download.json")
    if os.path.exists(json_download_path) and os.path.getsize(json_download_path) > 0:
        with open(json_download_path, "r") as f:
            json_data = f.read()
    else:
        json_data = "{}"
    if json_data != "{}":
        st.download_button(
            label="Download Bill JSON",
            data=json_data,
            file_name="bill_data.json",
            mime="application/json",
            icon=":material/download:",
            key="download_bill_json"
        )


if st.session_state.parsed_bill:
    import pandas as pd
    if isinstance(session_data["parsed_bill"], str):
        bill_data = json.loads(session_data["parsed_bill"])
    else:
        bill_data = session_data["parsed_bill"]
    flat_data = []
    for section, items in bill_data.items():
        if isinstance(items, list):
            for i in items:
                if isinstance(i, dict) and "key" in i and "value" in i:
                    flat_data.append({"Section": section, "Field": i["key"], "Value": i["value"]})
                else:
                    flat_data.append({"Section": section, "Field": str(i), "Value": ""})
        else:
            flat_data.append({"Section": section, "Field": "", "Value": str(items)})

    df = pd.DataFrame(flat_data)
    st.subheader("Parsed Bill Data")
    st.dataframe(df)

if user_input:
    st.session_state.bookena = False
    st.session_state.parsed_bill = False
    # --- Store image if uploaded ---
    ocr_text = ""
    if uploaded_image:
        print(f"image is uplaoded")
        # ---- Initialize OCR model asynchronously on startup ----
        if "ocr_model" not in st.session_state:
            async def load_ocr_model():
                return OptimizedOCRModel()
            st.session_state.ocr_model = asyncio.run(load_ocr_model())
            print(f"ocr model is loaded")

        temp_dir = tempfile.mkdtemp()
        img_path = os.path.join(temp_dir, uploaded_image.name)
        with open(img_path, "wb") as f:
            f.write(uploaded_image.read())
        session_data["images"].append({"filename": uploaded_image.name, "path": img_path})

        # OCR extraction
        ocr_model = st.session_state.ocr_model
        ocr_text = ocr_model.ocr_from_image(img_path, max_new_tokens=1500)
        print(f"ocr_text:{ocr_text}")
        session_data["ocr"].append({"filename": uploaded_image.name, "text": ocr_text})

    # --- Store user query ---
    session_data["chat"].append({"role": "user", "text": user_input})

    # --- Intent detection ---
    intent_result = process_input(user_input)
    intent = intent_result["intent"]
    prompt_template = intent_result["prompt"]
    session_data["chat"].append({"role": "user", "text": user_input})

    # --- Build prompt context ---
    all_chat_text = "\n".join([f"{m['role']}: {m['text']}" for m in session_data["chat"]])
    all_ocr_text = "\n".join([o["text"] for o in session_data["ocr"]]) or ocr_text
    prompt = prompt_template.format(all_ocr_text=all_ocr_text, all_chat_text=all_chat_text)
    print(f"intent : {intent} ; prompt : {prompt}")

    # --- Call model based on intent ---
    if intent in ["bill_parsing", "data_export"]:
        response = asyncio.run(query_ollama(prompt, bill_mode=True))
        print(f"response : {response}")
#         response = """
# {"header": [{"key": "Invoice Number", "value": "FABZDC2100864029"}, {"key": "Order ID", "value": "OD19157701078366432000"}, {"key": "Order Date", "value": "31-08-2020"}, {"key": "Invoice Date", "value": "31-08-2020"}, {"key": "GSTIN", "value": "29AAICA4872D1ZK"}, {"key": "PAN", "value": "AAICA4872D"}, {"key": "CIN", "value": "U51100DL2010PTC202600"}], "seller_info": [{"key": "Seller Name", "value": "Tech-Connect Retail Private Limited"}, {"key": "Ship-from Address", "value": "5, No.10,236,239 and 283, Maha Industrial Area, Heshkote Village Lakhur Hubli Muhar Taluk, Kolar, Karnataka, 563101, Bangalore, Karnataka, India - 563101, IN-KA"}], "buyer_info": [{"key": "Bill To", "value": "Amrishbhai Patel"}, {"key": "Bill Address", "value": "03 - Venice Bungalows,Near SP Ring Road,Behind Rainforest Bungalows, Nana Cottages Ahmedabad 382330 Gujarat Phone: xxxxxxxxxx"}, {"key": "Ship To", "value": "Amrishbhai Patel"}, {"key": "Ship Address", "value": "03 - Venice Bungalows,Near SP Ring Road,Behind Rainforest Bungalows, Nana Cottages Ahmedabad 382330 Gujarat Phone: xxxxxxxxxx"}], "totals": [{"key": "Total items", "value": "1"}, {"key": "Grand Total", "value": "42990.00"}], "line_items": [{"description": "Honor MagicBook 15 Ryzen", "quantity": "1", "unit_price": "42990.00", "total": "42990.00"}]}
# """
        session_data["chat"].append({"role": "assistant", "text": "Bill parsed successfully"})
        session_data["parsed_bill"] = response
        if isinstance(session_data["parsed_bill"], str):
            response = session_data["parsed_bill"].replace("'", '"')
        else:
            response = json.dumps(session_data["parsed_bill"])

    

        if intent == "data_export":
            print(f"after this intent == 'data_export': response : {response}")
                # Ensure parsed_bill is a JSON string
            json_data = response if isinstance(response, str) else json.dumps(response, indent=4)
            print(f"json_Data : {json_data}")
            # Write the JSON to a local file for reference
            try:
                    json_download_path = os.path.join(BASE_PATH, current_session, "json_download.json")
                    with open(json_download_path, "w") as f:
                        f.write(json_data)
                    print(f"Saved JSON to {json_download_path}")
            except Exception as e:
                    st.write(f"Error saving local JSON file: {e}")

            st.session_state.bookena = True
            st.session_state.parsed_bill = False
            
        elif intent == "bill_parsing":
            print(f"group : bill_parsing")
            print(f"response : {response}")
            # Display parsed bill JSON in tabular format
            st.session_state.bookena = False
            st.session_state.parsed_bill = True

    else:
        # In case of ChitChat or Information Extraction from OCR data we would use ollama - llama3 model local
        response = asyncio.run(query_ollama(prompt))
        session_data["chat"].append({"role": "assistant", "text": response})

    # --- Save session ---
    save_session_data(current_session, session_data)
    uploaded_image = None
    st.rerun()


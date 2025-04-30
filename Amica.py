import streamlit as st
import os
import google.generativeai as genai
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import pytesseract
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# --- Basic Styling ---
st.set_page_config(page_title="Amica 📚", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #F7F9FB; }
    h1    { text-align: center; color: #6C63FF; font-size: 42px; }
    </style>
""", unsafe_allow_html=True)

# --- Load environment & configure Gemini API ---
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# --- Ensure Tesseract is configured ---
TESS_PATH = r"E:\Colledge (CU)\Semester-8\Capstone Project\Apica-main\Tesseract\tesseract.exe"
if not os.path.isfile(TESS_PATH):
    raise FileNotFoundError(f"Tesseract binary not found at {TESS_PATH}")
pytesseract.pytesseract.tesseract_cmd = TESS_PATH

# --- History DataFrame (for appending) ---
conversation_df = pd.DataFrame(columns=["Timestamp", "Name", "UID", "User_Input", "Response"])

# --- PDF & FAISS Helpers ---
def get_pdf_text(pdf_file) -> str:
    text = ""
    reader = PdfReader(pdf_file)
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text += page_text
    return text

def get_text_chunks(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

def build_faiss_index(chunks: list[str]):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    store = FAISS.from_texts(chunks, embedding=embeddings)
    store.save_local("faiss_index")

def generate_quiz_from_pdf_content(pdf_text: str) -> str:
    prompt = f"""Create a 5-question multiple-choice quiz (with 4 options each and answer key) based on this content:
{pdf_text}
"""
    model = genai.GenerativeModel('gemini-2.0-flash-001')
    return model.generate_content(prompt).text

# --- LangChain QA Chain ---
def get_conversational_chain():
    template = """
You are Amica, an AI tutor. Use the provided context to answer clearly.

Context:
{context}

Question:
{question}

Answer:
"""
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", temperature=0.3)
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# --- Web + Personalization Handler ---
def handle_direct_query_web(query: str, name: str, uid: str, student_info: str) -> str:
    direct = genai.GenerativeModel('gemini-2.0-flash-001').generate_content(query).text
    user_hist = []
    if os.path.exists("conversation_history.csv"):
        hist = pd.read_csv("conversation_history.csv")
        user_hist = hist[hist.UID == uid].tail(5)[["User_Input", "Response"]].to_dict('records')

    personalization_prompt = f"""Personalize this answer:
- Name: {name}
- Student info: {student_info}
- Original question: "{query}"
- Knowledge-based answer: "{direct}"
- Last five conversation entries: {user_hist}

Produce a cohesive, personalized response."""
    chain = get_conversational_chain()
    enriched = chain({"input_documents": [], "question": personalization_prompt}, return_only_outputs=True)
    response = enriched["output_text"]

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conversation_df.loc[len(conversation_df)] = [ts, name, uid, query, response]
    conversation_df.to_csv(
        "conversation_history.csv",
        index=False,
        mode="a",
        header=not os.path.exists("conversation_history.csv")
    )
    return response

# --- PDF Q&A Handler ---
def handle_uploaded_query(query: str, name: str, uid: str) -> str:
    if not os.path.exists("faiss_index"):
        return "❗ Please process a PDF first."
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = store.similarity_search(query)
    chain = get_conversational_chain()
    out = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
    response = out["output_text"]

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conversation_df.loc[len(conversation_df)] = [ts, name, uid, query, response]
    conversation_df.to_csv(
        "conversation_history.csv",
        index=False,
        mode="a",
        header=not os.path.exists("conversation_history.csv")
    )
    return response

# --- PDF Solution Suggestions ---
def handle_pdf_solution(query: str, name: str, uid: str) -> str:
    if not os.path.exists("faiss_index"):
        return "❗ Please process a PDF first."
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = store.similarity_search(query)
    sol_prompt = f"Based on the provided context, suggest improved solutions for: {query}"
    chain = get_conversational_chain()
    out = chain({"input_documents": docs, "question": sol_prompt}, return_only_outputs=True)
    solution = out["output_text"]

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conversation_df.loc[len(conversation_df)] = [ts, name, uid, query, solution]
    conversation_df.to_csv(
        "conversation_history.csv",
        index=False,
        mode="a",
        header=not os.path.exists("conversation_history.csv")
    )
    return solution

# --- Image Q&A Handler ---
def handle_image_query(query: str, img_context: str, name: str, uid: str) -> str:
    prompt = f"""Given the following image context:
{img_context}

Answer the question: {query}
"""
    answer = genai.GenerativeModel('gemini-2.0-flash-001').generate_content(prompt).text

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conversation_df.loc[len(conversation_df)] = [ts, name, uid, query, answer]
    conversation_df.to_csv(
        "conversation_history.csv",
        index=False,
        mode="a",
        header=not os.path.exists("conversation_history.csv")
    )
    return answer

# --- Image Helpers ---
def pil_to_cv2(img: Image.Image) -> np.ndarray:
    rgb = img.convert("RGB")
    arr = np.array(rgb)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def segment_image_with_opencv(img: Image.Image):
    bgr = pil_to_cv2(img)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cnts

def describe_segments(cnts) -> list[str]:
    out = []
    for c in cnts:
        if cv2.contourArea(c) > 500:
            out.append("Detected a significant object or shape.")
    return out or ["No major objects detected."]

def recognize_text(img: Image.Image) -> str:
    bgr = pil_to_cv2(img)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return pytesseract.image_to_string(gray).strip()

def classify_image(img: Image.Image) -> str:
    model = MobileNetV2(weights='imagenet')
    rgb = img.convert("RGB").resize((224,224))
    arr = img_to_array(rgb)
    arr = np.expand_dims(arr, 0)
    arr = preprocess_input(arr)
    preds = model.predict(arr)
    decoded = decode_predictions(preds, top=3)[0]
    return ", ".join([f"{label}: {prob*100:.2f}%" for (_, label, prob) in decoded])

# --- Main App ---
def main():
    st.title("Amica 📚 - Intelligent Adaptive Learning Companion")

    # Session-state flags
    if "pdf_done" not in st.session_state:
        st.session_state.pdf_done = False
    if "img_up" not in st.session_state:
        st.session_state.img_up = False

    # Sidebar: developer credit (all three styles)
    st.sidebar.markdown("<small><em>Developed by Vivek, Hariom, Pranab, Himansh</em></small>", unsafe_allow_html=True)
    # st.sidebar.markdown(
    #     '<strong style="color:#6C63FF;">Developed by Vivek Kumar</strong>',
    #     unsafe_allow_html=True
    # )
    # st.sidebar.markdown("🎨👨‍💻 Developed by Vivek Kumar")

    # Sidebar: user info
    name = st.sidebar.text_input("Name")
    uid  = st.sidebar.text_input("UID")
    student_info = ""
    if st.sidebar.checkbox("Activate Profile"):
        student_info = st.sidebar.text_area("About You (goals, strengths…)")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "💬 Ask (Web + Personalized)",
        "📚 PDF & Quiz",
        "🖼️ Image Analysis",
        "🕑 Conversation History"
    ])

    # Tab 1: Web + Personalization
    with tab1:
        st.subheader("Ask Amica!!!")
        web_q = st.text_input("Your question:", key="web_q")
        if st.button("Ask Web", key="ask_web"):
            if not (name and uid):
                st.warning("Enter Name & UID first.")
            else:
                ans = handle_direct_query_web(web_q, name, uid, student_info)
                st.markdown(f"**You:** {web_q}")
                st.markdown(f"**Amica:** {ans}")

    # Tab 2: PDF Q&A, Solutions & Quiz
    with tab2:
        st.subheader("Upload & Process PDF")
        pdf_file = st.file_uploader("Choose a PDF", type="pdf", key="pdf_u")
        if pdf_file and not st.session_state.pdf_done:
            if st.button("Process PDF"):
                raw_text = get_pdf_text(pdf_file)
                st.session_state.pdf_text = raw_text
                chunks = get_text_chunks(raw_text)
                build_faiss_index(chunks)
                st.session_state.pdf_done = True
                st.success("PDF indexed!")
        if st.session_state.pdf_done:
            pdf_q = st.text_input("Ask about PDF:", key="pdf_q")
            if st.button("Ask PDF", key="ask_pdf"):
                ans_pdf = handle_uploaded_query(pdf_q, name, uid)
                st.markdown(f"**You:** {pdf_q}")
                st.markdown(f"**Amica:** {ans_pdf}")
            sol_q = st.text_input("What solution would you like Amica to propose?", key="pdf_sol_q")
            if st.button("Get PDF Solutions", key="ask_pdf_sol"):
                sol = handle_pdf_solution(sol_q, name, uid)
                st.markdown(f"**Amica's Solution:** {sol}")
            if st.button("Generate Quiz from PDF", key="quiz_pdf"):
                quiz = generate_quiz_from_pdf_content(st.session_state.pdf_text)
                st.subheader("🧩 Quiz:")
                st.write(quiz)

    # Tab 3: Image Analysis + Q&A
    with tab3:
        st.subheader("Image Analysis")
        img_file = st.file_uploader("Upload image", type=["png","jpg","jpeg"], key="img_u")
        if img_file is None:
            st.session_state.img_up = False
        elif not st.session_state.img_up and st.button("Analyze Image"):
            st.session_state.img_up = True
            st.session_state.img_obj = Image.open(img_file)

        if st.session_state.img_up:
            img_obj = st.session_state.img_obj
            st.image(img_obj, use_column_width=True)

            cnts = segment_image_with_opencv(img_obj)
            descs = describe_segments(cnts)
            st.markdown("**Detected:**")
            for d in descs:
                st.write(f"- {d}")

            text = recognize_text(img_obj)
            if text:
                st.markdown("**Recognized Text:**")
                st.write(text)

            cls = classify_image(img_obj)
            st.markdown("**Classification:**")
            st.write(cls)

            img_context = f"Segments: {descs}\nRecognized text: {text}\nClassification: {cls}"
            img_q = st.text_input("Ask Amica about this image:", key="img_q")
            if st.button("Ask Image", key="ask_image"):
                img_ans = handle_image_query(img_q, img_context, name, uid)
                st.markdown(f"**You:** {img_q}")
                st.markdown(f"**Amica:** {img_ans}")


    # Tab 4: Conversation History
    with tab4:
        st.subheader("Conversation History")
        if os.path.exists("conversation_history.csv"):
            df = pd.read_csv("conversation_history.csv")
            st.dataframe(df[df.UID == uid])
        else:
            st.info("No history yet.")

if __name__ == "__main__":
    main()

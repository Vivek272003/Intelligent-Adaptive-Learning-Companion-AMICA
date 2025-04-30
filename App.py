import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import pandas as pd
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from datetime import datetime
from PIL import Image

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
conversation_df = pd.DataFrame(columns=["Timestamp", "Name","UID","User_Input", "Response"])
feedback=pd.DataFrame(columns=["UID","Name", "Feedback"])

def get_gemini_response(input,image):
    model=genai.GenerativeModel('models/gemini-2.0-flash-exp-image-generation')
    if input!="":
        response = model.generate_content([input,image])
    else:
        response = model.generate_content(image)
    return response.text
        

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings )
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Your name is "Curio". user will input his details, please answer keeping the details {AboutStudent} in mind.
    Answer in the format of a teacher. You have to explain in a way so that the student understands well the concept ans is clear regarding the same.
    Name: \n {name} \n
    Context:\n {context}?\n
    Question: \n{question}\n
    AboutStudent: \n{AboutStudent}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context","question","name","AboutStudent"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain



def take_quiz(name,uid):

    question_prompt='''Generate a Short question based on the {context}  of the student understanding and the previous chats that has occured. 
    It has to be a short question which is easy to evaluate.
    # '''
    chain=get_conversational_chain()
    df=pd.read_csv('conversation_history.csv')
    df.columns=['Timestamps','Name','UID','User_Input','Response']
    df=df[df.UID==uid]
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    history=df['User_Input'].tolist()+df['Response'].tolist()
    docs = new_db.similarity_search(history)
    result = chain({"input_documents": docs, "question": question_prompt, "name": name, "AboutStudent": []}, return_only_outputs=True)
    question=result['output_text']
    st.write(question)
    answer=[]
    answer.append(st.text_input("Give an answer"))
    st.write("--------")

    evaluation_parameter=f'''
    Evaluate in short andstudent based on  {answer[0]} on the questions {question} returning score(ranging from 0 to 10) and detailed performance analysis.
    Highlight missed questions with corrective explanations.
    Offer personalized feedback addressing knowledge gaps, strengths, and learning opportunities.
    '''    
    evaluation= chain({"input_documents": [], "question": evaluation_parameter, "name": name, "AboutStudent": answer[0]}, return_only_outputs=True)
    feedback.loc[len(feedback)] = [uid,name,evaluation]
    feedback.to_csv("feedback_history.csv", index=False, mode='a',  header=not os.path.exists("feedback_history.csv"))
    
def student_info(name):
    # Generate greeting message using language model
    with st.sidebar:
        answers=[]
        model = get_conversational_chain()
        on=st.toggle("Profile Playground")
        if on:
            prompt = """
            Generate a good and descriptive user greeting given the following inputs: 
            Ask some questions related to capturing student information like Name, Age, Gender, Learning History:, Learning Styles, Accessibility Needs, Areas of Interest, Preferred Modalities
            Name: \n {name} \n
            make sure the questions are concise.
            Answer: 
            """
            # response = model(prompt)
            response = model({"input_documents": [], "question": prompt, "name": name, "AboutStudent":[]}, return_only_outputs=True)
            st.write(response["output_text"])
            answers=st.text_input("Answer here")
    
    greet="Generate a greeting message as per the information you under about the user. If user information is not available, simple greet generally"
    greeting = model({"input_documents": [], "question": greet, "name":name, "AboutStudent":answers})
    st.write(greeting["output_text"])
    return answers
    
    

def user_input(user_question, name,uid, answers):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings,  allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question, "name": name, "AboutStudent": answers}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])


    if st.button("Regenerate"):
            with st.spinner("regenerating...."):
                chain = get_conversational_chain()
                response = chain({"input_documents": docs, "question": user_question, "name": name, "context":new_db, "AboutStudent": answers}, return_only_outputs=True)
                st.write("Regenerated Reply: ", response["output_text"])
                
    conversation_df.loc[len(conversation_df)] = [timestamp, name, uid,user_question, response["output_text"]]
    conversation_df.to_csv("conversation_history.csv", index=False, mode='a',  header=not os.path.exists("conversation_history.csv"))
    return answers
    
def main():
    st.set_page_config("Chat PDF")
    with st.sidebar:
        st.write("*Developed by Vivek Kumar*")
        
    
    st.header("Curio \n*Learn your way, At your pace*")
    
    with st.sidebar:
        st.title("Details")
        name = st.text_input("Enter your name")
        uid=st.text_input("Enter your UID")
        on=st.toggle("Quizzie")
        if on:
            take_quiz(name,uid)
            
    with st.sidebar:
        on=st.toggle("Upload Document")
        if on:
            st.title("Menu:")
            pdf_docs = st.file_uploader("Upload your Document Files to provide context", accept_multiple_files=True)
            if st.button("Submit & Process"):
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")
            
            image_uploader=st.file_uploader("Upload Image")
            input=st.text_input("Input Prompt: ",key="input")
            image=""   
            if image_uploader is not None:
                image = Image.open(image_uploader)
                st.image(image, caption="Uploaded Image.", use_column_width=True)
            
            if st.button("Tell me about the image"):
                response=get_gemini_response(input,image)
                st.write(response)
 
    answers=student_info(name)
    
    user_question = st.text_input("Ask a Question")
    if user_question:
        user_input(user_question, name,uid, answers)
        df=pd.read_csv('conversation_history.csv')
        df.columns=['Timestamps','Name','UID','User_Input','Response']
        df=df[df.UID==uid]
        for i in range(len(df)):
            st.write("---------")
            st.write(f"*Time*: {df['Timestamps'].iloc[len(df)-i-1]}")
            st.write(f"*User*: {df['User_Input'].iloc[len(df)-i-1]}")
            st.write(f"*Response*: {df['Response'].iloc[len(df)-i-1]}")
    
     

                        


if __name__ == "__main__":
    main()

from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import streamlit as st

def main():
    load_dotenv()
    st.set_page_config(page_title="Homework GPT")
    st.header("Homework GPT")

    #Upload file
    pdf = st.file_uploader("Upload your pdf", type="pdf")

    #Extract the text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        #Split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        #Create embeddings
        embeddings = OpenAIEmbeddings()

        #Similarity Search
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        #Show the user input
        user_question = st.text_input("What is your question? ")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)

            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")

            #See cost of each use
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)
                print(cb)

            st.write(response)


    
if __name__ =='__main__':
    main()

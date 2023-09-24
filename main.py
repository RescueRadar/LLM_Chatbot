from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import DirectoryLoader

from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(temperature=0,model_name="gpt-3.5-turbo")

import os
os.environ["OPENAI_API_KEY"]= os.getenv("API_KEY")

pdf_loader = DirectoryLoader(os.getenv("DIR_PATH"), glob="**/*.pdf")
txt_loader = DirectoryLoader(os.getenv("DIR_PATH"), glob="**/*.txt")
word_loader = DirectoryLoader(os.getenv("DIR_PATH"), glob="**/*.docx")

loaders = [pdf_loader, txt_loader , word_loader]
documents = []
for loader in loaders:
    documents.extend(loader.load())

text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
documents = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings)

qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True)

import gradio as gr
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")
    chat_history = []
    def user(query, chat_history):
        print("User query:", query)
        print("Chat history:", chat_history)

        chat_history_tuples = []
        for message in chat_history:
            chat_history_tuples.append((message[0], message[1]))
        result = qa({"question": query, "chat_history": chat_history_tuples})
        chat_history.append((query, result["answer"]))
        print("Updated chat history:", chat_history)
        return gr.update(value=""), chat_history
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False)
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(debug=True,share=True
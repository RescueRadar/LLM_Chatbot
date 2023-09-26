import streamlit as st
import streamlit.components.v1 as components
import os
from dotenv import load_dotenv
load_dotenv()
def perform_action(option):
    if option == 'Agency Query':
        st.write('Here is your query chatbot')
        from langchain.embeddings.openai import OpenAIEmbeddings
        from langchain.vectorstores import Chroma
        from langchain.text_splitter import CharacterTextSplitter
        from langchain.chains import ConversationalRetrievalChain
        

        os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

        from langchain.chat_models import ChatOpenAI
        llm = ChatOpenAI(temperature=0,model_name="gpt-3.5-turbo")

        from langchain.document_loaders import DirectoryLoader

        txt_loader = DirectoryLoader(os.getenv('DIR_PATH'), glob="**/*.txt")
        word_loader = DirectoryLoader(os.getenv('DIR_PATH'), glob="**/*.docx")

        loaders = [txt_loader , word_loader]
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
        components.iframe("http://127.0.0.1:7860", width=800,height=600,scrolling=True)
        if __name__ == "__main__":
            demo.launch(debug=True,share=True)

    elif option == 'General Query':
        st.write('Here is your query chatbot')

        import openai 
        import gradio  
        openai.api_key = os.getenv('OPENAI_API_KEY')
        messages=[{"role":"system","content":"you are a rescue agency"}]  

        def CustomChatGPT(user_input):
            messages.append({"role": "user", "content": user_input})
            response = openai.ChatCompletion.create(
                model = "gpt-3.5-turbo",
                messages = messages
            )
            ChatGPT_reply = response["choices"][0]["message"]["content"]
            messages.append({"role": "assistant", "content": ChatGPT_reply})
            return ChatGPT_reply

        demo = gradio.Interface(fn=CustomChatGPT, inputs = "text", outputs = "text", title = "Calamity chatbot")
        components.iframe("http://127.0.0.1:7862", width=800, height=600, scrolling=True)
        demo.launch(share=True)

st.title('Select Chatbot type')

option_selected = st.radio('Select an option:', ('Agency Query', 'General Query'))

if st.button('Perform Action'):
    perform_action(option_selected)
    
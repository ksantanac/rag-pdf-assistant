import streamlit as st
import os

from dotenv import load_dotenv

from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai.chat_models import ChatOpenAI
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate

# Carrega o arquivo .env
load_dotenv()

# Configuração da página
st.set_page_config(page_title="Chat com PDFs 📚", layout="wide")
st.title("Chat com PDFs usando RAG 🤖📖")

# Inicializa histórico de mensagens
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Sidebar: PDFs carregados
st.sidebar.header("📂 Documentos carregados")
uploaded_files = st.sidebar.file_uploader(
    "Faça upload dos seus PDFs", type="pdf", accept_multiple_files=True
)

caminhos = []
if uploaded_files:
    pasta_pdf = "data"
    os.makedirs(pasta_pdf, exist_ok=True)
    
    for file in uploaded_files:
        caminho = os.path.join(pasta_pdf, file.name)
        with open(caminho, "wb") as f:
            f.write(file.read())
        caminhos.append(caminho)
        st.sidebar.write(f"✅ {file.name}")

# Se houver PDFs carregados
if caminhos:
    # Carregar páginas dos PDFs
    paginas = []
    for caminho in caminhos:
        loader = PyPDFLoader(caminho)
        paginas.extend(loader.load())

    # Split em chunks
    recur_split = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    documents = recur_split.split_documents(paginas)

    for i, doc in enumerate(documents):
        doc.metadata['source'] = doc.metadata.get('source', 'pdf')
        doc.metadata['doc_id'] = i

    # Criar Vectorstore (persistente)
    diretorio = "data/chat_retrieval_db"
    embeddings_model = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings_model,
        persist_directory=diretorio
    )

    # LLM
    chat = ChatOpenAI(model="gpt-4o-mini")

    # Prompt
    chain_prompt = PromptTemplate.from_template(
        """
        Você é um assistente especializado em responder perguntas com base nos documentos fornecidos.

        Instruções:
        - Use apenas o contexto abaixo para responder.
        - Se a resposta não estiver clara no contexto, diga: "Não encontrei essa informação nos documentos."
        - Seja direto e conciso (máximo 3 frases).
        - Se possível, destaque palavras-chave importantes da resposta.
        - Não invente informações que não estejam no contexto.

        Contexto: {context}

        Pergunta: {question}

        Resposta:
        """
    )

    # Criar chain
    chat_chain = RetrievalQA.from_chain_type(
        llm=chat,
        retriever=vectordb.as_retriever(search_type="mmr"),
        chain_type_kwargs={"prompt": chain_prompt},
        return_source_documents=True
    )

    # Botão para exportar histórico
    if st.sidebar.button("📥 Exportar conversa"):
        historico = "\n\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.messages])
        st.sidebar.download_button("Baixar histórico", historico, "chat.txt")

    # Exibe histórico no chat
    for msg in st.session_state.messages:
        role = msg["role"]
        avatar = "👤" if role == "user" else "🤖"
        with st.chat_message(role, avatar=avatar):
            st.markdown(msg["content"])

    # Caixa de input estilo chat
    if pergunta := st.chat_input("Digite sua pergunta..."):
        # Mostra pergunta do usuário
        st.session_state.messages.append({"role": "user", "content": pergunta})
        with st.chat_message("user", avatar="👤"):
            st.markdown(pergunta)

        # Gera resposta
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("💭 Pensando na resposta..."):
                resposta = chat_chain.invoke({"query": pergunta})
                resposta_texto = resposta["result"]
                st.markdown(resposta_texto)

                # Mostra fontes
                with st.expander("📂 Fontes consultadas"):
                    for doc in resposta["source_documents"]:
                        st.write(f"Arquivo: {doc.metadata['source']}")
                        st.write(doc.page_content[:300] + "...")
                        st.markdown("---")

        # Salva resposta no histórico
        st.session_state.messages.append({"role": "assistant", "content": resposta_texto})

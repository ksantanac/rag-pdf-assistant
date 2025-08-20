# Chat com PDFs usando RAG 🤖📖

Um projeto de **Chat baseado em PDFs** utilizando **RAG (Retrieval-Augmented Generation)** e **LangChain**.  
O usuário pode enviar múltiplos PDFs, fazer perguntas sobre o conteúdo e receber respostas baseadas nos documentos.

O projeto utiliza **Streamlit** para interface web e integra **OpenAI GPT-4o-mini** para respostas.

---

## 🔹 Funcionalidades

- Upload de múltiplos PDFs.
- Processamento e divisão dos documentos em chunks.
- Criação de **VectorStore persistente** com embeddings.
- Perguntas e respostas contextuais usando RAG.
- Histórico de conversas com **chat estilo mensageria**.
- Avatares e estilização das mensagens:
  - 🤖 Assistente
  - 👤 Usuário
- Exportação do histórico em `.txt`.
- Visualização dos PDFs carregados no sidebar.
- Fonte de cada resposta exibida com preview do conteúdo.

---

## 🔹 Tecnologias utilizadas

- [Python 3.10+](https://www.python.org/)
- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [OpenAI GPT](https://platform.openai.com/)
- [ChromaDB](https://www.trychroma.com/) para armazenamento de embeddings
- [dotenv](https://pypi.org/project/python-dotenv/) para gerenciar a API key

---

## 🔹 Estrutura do Projeto

```text
rag-pdf-assistant/
│
├─ data/                      
├─ app.py                       
├─ requirements.txt             
└─ README.md
```

---

## 🔹 Instalação

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/rag-pdf-assistant.git
cd rag-pdf-assistant

# Crie e ative um ambiente virtual
python -m venv env

# Windows
env\Scripts\activate

# Linux / Mac
source env/bin/activate

# Instale as dependências
pip install -r requirements.txt

# Configure sua API key da OpenAI em um arquivo .env
echo 'OPENAI_API_KEY="sua_api_key_aqui"' > .env
```

---

## 🔹 Uso

```bash
# Execute a aplicação
streamlit run app.py
```


## 📫 **Contato**: [kauesantana_13@hotmail.com]
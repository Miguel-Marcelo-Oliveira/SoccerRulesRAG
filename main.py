# Importações básicas
import os
from dotenv import load_dotenv

# Loader de documentos PDF
from langchain_community.document_loaders import PyPDFLoader

# Divisão de texto em blocos
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings (via API do Cloudflare Workers AI)
from langchain_cloudflare.embeddings import CloudflareWorkersAIEmbeddings

# Banco vetorial
from langchain_community.vectorstores import Chroma

# LLM (via Ollama)
from langchain_ollama import ChatOllama

# Cadeia RAG
from langchain.chains import RetrievalQA

# Carrega as variáveis do .env para o ambiente
load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "https://ollama.com")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:120b")
CF_ACCOUNT_ID = os.getenv("CF_ACCOUNT_ID")
CF_AI_API_TOKEN = os.getenv("CF_AI_API_TOKEN")

if not OLLAMA_API_KEY:
    raise ValueError("OLLAMA_API_KEY não encontrada. Verifique seu arquivo .env")

if not CF_ACCOUNT_ID or not CF_AI_API_TOKEN:
    raise ValueError("CF_ACCOUNT_ID ou CF_AI_API_TOKEN não encontrados. Verifique seu arquivo .env")

# Inicializar os Embeddings via API do Cloudflare Workers AI (bge-m3, multilíngue)
# Free tier: 10.000 Neurons/dia, ~3.000 requisições/min — não precisa de rate limiting manual
embeddings = CloudflareWorkersAIEmbeddings(
    account_id=CF_ACCOUNT_ID,
    api_token=CF_AI_API_TOKEN,
    model_name="@cf/baai/bge-m3"
)

PERSIST_DIR = "./chroma_regras_futebol"

# Indexa o PDF apenas se o banco ainda não existir (evita duplicar chunks)
if os.path.exists(PERSIST_DIR):
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )
else:
    # Carregando o documento PDF
    CAMINHO_PDF = "regras_futebol.pdf"

    loader = PyPDFLoader(CAMINHO_PDF)
    documents = loader.load()
    # print(len(documents))

    # Dividindo os documentos em chunks menores
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(documents)
    # print(len(chunks))

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )

# Criar o retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Inicializando o modelo de linguagem
llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_HOST,
    headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"}
)

# Criar a cadeia RAG
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Loop de perguntas
while True:
    pergunta = input("Insira sua pergunta (ou 'sair' para encerrar): ").strip()

    if not pergunta:
        continue

    if pergunta.lower() == "sair":
        print("Até a próxima!")
        break

    # Executa a pergunta no agente RAG
    resposta = qa_chain.invoke({"query": pergunta})

    print("Pergunta:", pergunta)
    print("\nResposta do agente:", resposta["result"])
    print("\nTrechos utilizados como contexto:\n")

    for i, doc in enumerate(resposta["source_documents"], start=1):
        print(f"--- Trecho {i} ---")
        print(f"Fonte: {doc.metadata.get('source', 'Documento desconhecido')}")
        print(f"Página: {doc.metadata.get('page', 'N/A')}")
        print("Conteúdo:")
        print(doc.page_content)
        print("\n")

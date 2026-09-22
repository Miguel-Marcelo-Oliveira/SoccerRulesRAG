# Agente RAG — Regras do Jogo (IFAB)

Agente de perguntas e respostas (chatbot) sobre o PDF **Regras do Jogo 2026/27** da IFAB/FIFA, construído com **RAG (Retrieval-Augmented Generation)** usando LangChain, ChromaDB, Ollama e/ou Cloudflare Workers AI.

Faça perguntas em linguagem natural (ex.: *"O que acontece se o goleiro sair da linha no pênalti?"*) e o agente recupera os trechos mais relevantes do documento, gera uma resposta e mostra as fontes com número da página.

---

## Estrutura do projeto

| Arquivo              | Descrição                                                                              |
|----------------------|----------------------------------------------------------------------------------------|
| `main.py`            | Script principal — embeddings via API do Cloudflare Workers AI e LLM via API do Ollama |
| `regras_futebol.pdf` | Documento base: Regras do Jogo 2026/27 (IFAB)                                          |
| `requirements.txt`   | Dependências do projeto                                                                |
| `.env.example`       | Modelo de variáveis de ambiente (copiar para `.env`)                                   |

---

## Como funciona

```
regras_futebol.pdf
      │
      ▼
PyPDFLoader (carrega o PDF)
      │
      ▼
RecursiveCharacterTextSplitter (chunks de 500 caracteres, overlap de 100)
      │
      ▼
Embeddings (Cloudflare Workers AI)
      │
      ▼
ChromaDB (banco vetorial persistido em disco — indexado apenas na 1ª execução)
      │
      ▼
Retriever (busca os 3 chunks mais similares à pergunta)
      │
      ▼
ChatOllama (LLM gera a resposta com base nos trechos recuperados)
      │
      ▼
Resposta + fontes (páginas utilizadas)
```

---

## Como executar

### 1. Pré-requisitos

- **Python 3.10+**

> O LLM (que gera as respostas) é sempre acessado via Ollama — por padrão pela nuvem Ollama, 
> usando o modelo `gpt-oss:120b`. Isso exige uma **API key do Ollama**.

> O modelo que gera os embeddings (converte os textos em vetores numéricos) é sempre acessado via Cloud Flare,
> usando o modelo `@cf/baai/bge-m3`. Isso exige o ID e um TOKEN da conta.

### 2. Instalação

```bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
# venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

### 3. Configurar variáveis de ambiente

```bash
cp .env.example .env
```

Preencha o `.env`:

```env
OLLAMA_API_KEY=COLE_AQUI_SUA_API_KEY
OLLAMA_MODEL=gpt-oss:120b
CF_ACCOUNT_ID=COLE_AQUI_O_ID_DE_SUA_CONTA_CLOUDFLARE
CF_AI_API_TOKEN=COLE_AQUI_SEU_TOKEN
```

### 4. Rodar o agente

```bash
python main.py
```

### 5. Fazer perguntas

```
Insira sua pergunta (ou 'sair' para encerrar): Quantos jogadores no mínimo uma equipe precisa ter?
```

Digite `sair` para encerrar.

---

## Variáveis de ambiente

| Variável          | Descrição                                                              |
|-------------------|------------------------------------------------------------------------|
| `OLLAMA_API_KEY`  | Chave de API do Ollama (acesso via [ollama.com](`https://ollama.com`)) |
| `OLLAMA_MODEL`    | Modelo de linguagem (padrão: `gpt-oss:120b`)                           |
| `OLLAMA_HOST`     | Host do LLM                                                            |
| `CF_ACCOUNT_ID`   | ID da conta Cloudflare                                                 |
| `CF_AI_API_TOKEN` | Token da API de IA do Cloudflare                                       |

---

## Bancos vetoriais

Os índices são salvos em disco e reutilizados nas execuções seguintes (o PDF só é indexado uma vez):

- `chroma_regras_futebol/` — índice salvo após execução do código


Para **reindexar** (ex.: após trocar o modelo de embeddings ou alterar o PDF), basta apagar a pasta correspondente:

```bash
rm -rf chroma_regras_futebol
```

---

## Dependências principais

- `langchain` / `langchain-community` / `langchain-text-splitters` — orquestração e divisão de documentos
- `langchain-ollama` — integração com LLM e embeddings do Ollama
- `langchain-cloudflare` — embeddings do Cloudflare Workers AI (`@cf/baai/bge-m3`)
- `chromadb` — banco vetorial
- `pypdf` — leitura do PDF
- `python-dotenv` — carregamento do `.env`

---

## Ideias de perguntas

- "Quanto tempo dura um jogo de futebol?"
- "Quais as medidas do campo em jogos internacionais?"
- "O que acontece se o goleiro pegar a bola de volta no arremesso lateral?"
- "Em que situações um jogador leva cartão amarelo?"
- "Como funciona o desempate na disputa de pênaltis?"

---

## Observações

- O retriever busca os **3 trechos mais relevantes** (`k=3`) por pergunta.
- As respostas são geradas apenas com base no conteúdo do PDF (chain type `stuff`).
- O projeto utiliza as **Regras 2026/27** da IFAB.

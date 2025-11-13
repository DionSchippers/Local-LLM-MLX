import re
import fitz
import faiss
import torch
from sentence_transformers import SentenceTransformer
from mlx_lm import load, generate

PDF_PATH = "./datasets/roman_empire_dataset.pdf"
MODEL_NAME = "mlx-community/gpt-oss-20b-MXFP4-Q8"
EMBED_MODEL = "paraphrase-multilingual-mpnet-base-v2"
MARKER = "<|end|><|start|>assistant<|channel|>final<|message|>"
DEFAULT_TOKENS = 1500
TOKEN_LIMIT = 5000
INTRO_PROMPT = "You are a history expert specializing in the Roman Empire. Use the provided context to answer questions about the Roman Empire in detail and with historical accuracy. If you can't find the answer in the context, try and formulate an answer, but state that it is not provided in the context. Always answer in the same language as the user."
CHAT_HISTORY = []
CHAT_HISTORY_TEXT = ""
MODEL = None
TOKENIZER = None
CHUNKS = []
EMBEDDER = None
INDEX = None


def initialize_model():
    global TOKENIZER, MODEL, CHUNKS, EMBEDDER, INDEX
    MODEL, TOKENIZER = load(MODEL_NAME)
    generate(MODEL, TOKENIZER, "Hello", max_tokens=1)

    document = fitz.open(PDF_PATH)
    text = "".join(page.get_text() for page in document)
    clean_text = re.sub(r"<\|.*?\|>", "", text)
    CHUNKS = re.split(r"\n\s*\n", clean_text)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    EMBEDDER = SentenceTransformer(EMBED_MODEL, device=device)
    vectors = EMBEDDER.encode(CHUNKS, convert_to_numpy=True)
    dimensions = vectors.shape[1]
    INDEX = faiss.IndexFlatL2(dimensions)
    INDEX.add(vectors)


def convert_chat_history_to_text():
    global CHAT_HISTORY_TEXT
    for chat in CHAT_HISTORY:
        if len(CHAT_HISTORY_TEXT) == 0:
            CHAT_HISTORY_TEXT = chat
        else:
            CHAT_HISTORY_TEXT = "\n".join(CHAT_HISTORY)


def remove_first_messages():
    global CHAT_HISTORY
    CHAT_HISTORY = CHAT_HISTORY[2:]


def calculate_token_length(query, context):
    text = f"{INTRO_PROMPT}\n{CHAT_HISTORY_TEXT}\n{query}\n{context}"
    tokens = TOKENIZER.encode(text)
    if len(tokens) > TOKEN_LIMIT:
        remove_first_messages()
        convert_chat_history_to_text()
        calculate_token_length(query, context)


def embed_query(text):
    return EMBEDDER.encode([text], convert_to_numpy=True)


def retrieve_context(query: str, k: int = 3, max_chars: int = 2500):
    query_vectors = embed_query(query)
    distances, indices = INDEX.search(query_vectors, k)
    selected = [CHUNKS[i][:max_chars] for i in indices[0]]
    return selected


def build_prompt(query, context):
    return f"""
{INTRO_PROMPT}

Context:
{context}

Conversation history:
{CHAT_HISTORY_TEXT}

Current question:
{query}

Answer:
"""


def get_answer(query):
    global CHAT_HISTORY
    convert_chat_history_to_text()
    context_blocks = retrieve_context(query)
    context = "\n\n".join(context_blocks)
    calculate_token_length(query, context)
    prompt = build_prompt(query, context)
    result = generate(
        MODEL,
        TOKENIZER,
        prompt,
        max_tokens=DEFAULT_TOKENS,
    )
    answer = result.strip().split(MARKER, 1)[1].strip()
    CHAT_HISTORY.append(f"User: {query}")
    CHAT_HISTORY.append(f"Assistant: {answer}\n")
    return answer


def run_roman_expert():
    print("🧠 Loading LLM...")
    initialize_model()

    print("\n✅ Conversator is ready!")
    print("Ask questions (type `quit` to exit)\n")

    while True:
        query = input("❓ > ")
        if query.strip().lower() in ("quit", "exit"):
            print("👋 Goodbye!")
            break

        if not query.strip():
            continue

        answer = get_answer(query)
        print("\n✅ Answer:\n", answer, '\n')

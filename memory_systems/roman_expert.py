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

# =========================
# Step 1: Initialize LLM model and Read the PDF document
# =========================
# Load the specified LLM model and tokenizer using the `mlx_lm` library.
# This so we can use all this later, but it isn't directly loaded once main gets run.
# We also read the PDF document containing information about the Roman Empire,
# extract the text, clean it up, and split it into chunks for embedding and indexing.
# =========================
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

# =========================
# Step 2: History management functions
# =========================
# We need to convert the array of chat history into a single string. This so it can be used in the prompt.
# We also need to manage the token length of the conversation history to ensure it stays within limits as a LLM has a max token limit.
# We want to remove the oldest messages when the limit is exceeded.
# Because the chat messages are stored as an array with questions and answers alternating, we remove the first two messages (one question and one answer) at a time.
# If this isn't enough, we recursively call the function until the token length is within limits.
# =========================
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

# =========================
# Step 3: Retrieve context & Build prompt
# =========================
# Make the Query into vectors.
# Search the FAISS index for the most relevant chunks.
# Build a prompt that includes the retrieved context and the user's query.
# =========================
def embed_query(text):
    return EMBEDDER.encode([text], convert_to_numpy=True)


def retrieve_context(query: str, k: int = 3, max_chars: int = 2500):
    query_vectors = embed_query(query)
    distances, indices = INDEX.search(query_vectors, k)
    selected = [CHUNKS[i][:max_chars] for i in indices[0]]
    return selected

# =========================
# Step 4: Build the prompt and get answer
# =========================
# We build the prompt by combining the intro prompt, the context, the conversation history, and the current question.
# It is important to keep a little room with your token limit as the generic text in the prompt is not calculated.
# This unfortunately is very difficult as the chat history can vary in length.
# After building the prompt, we generate an answer using the LLM model.
# When the answer is received, we update the chat history with the new question and answer.
# =========================
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

# =========================
# Step 5: Call all functions to get answer
# =========================
# We call all the functions to get the final answer from the LLM.
# We do this by first converting the chat history to text,
# retrieving the relevant context for the query,
# calculating the token length to ensure it fits within limits,
# building the prompt,
# generating the answer using the LLM,
# and finally updating the chat history with the new question and answer.
# =========================
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

# =========================
# Step 6: Run the llm
# =========================
# This function runs the main loop of the llm.
# It initializes the model, then enters a loop where it prompts the user for input,
# gets the answer from the LLM, and prints the answer.
# =========================
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

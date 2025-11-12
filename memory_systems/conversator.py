from mlx_lm import load, generate

from rag_systems.nederland_expert import chunks

PDF_PATH = "./datasets/nederland.pdf"
MODEL_NAME = "mlx-community/Llama-3.2-3B-Instruct"
DEFAULT_TOKENS = 1500
TOKEN_LIMIT = 300

INTRO_PROMPT = "You are a friendly and helpful conversational assistant. Always answer in the same language as the user."

CHAT_HISTORY = []
CHAT_HISTORY_TEXT = ""
MODEL = None
TOKENIZER = None


def initialize_model():
    global TOKENIZER, MODEL
    MODEL, TOKENIZER = load(MODEL_NAME)


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


def calculate_token_length(query):
    tokens = TOKENIZER.encode(INTRO_PROMPT) + TOKENIZER.encode(
        CHAT_HISTORY_TEXT) + TOKENIZER.encode(query)
    if len(tokens) > TOKEN_LIMIT:
        remove_first_messages()
        convert_chat_history_to_text()
        calculate_token_length(query)


def build_prompt(query):
    return f"""
{INTRO_PROMPT}

Conversation history:
{CHAT_HISTORY_TEXT}

Current question:
{query}

Answer:
"""


def get_answer(query):
    global CHAT_HISTORY
    convert_chat_history_to_text()
    calculate_token_length(query)
    prompt = build_prompt(query)
    result = generate(
        MODEL,
        TOKENIZER,
        prompt,
        max_tokens=DEFAULT_TOKENS,
    )
    answer = result.strip()
    CHAT_HISTORY.append(f"User: {query}")
    CHAT_HISTORY.append(f"Assistant: {answer}\n")

    return answer


def run_conversator():
    print("🧠 Loading LLM...")
    initialize_model()

    print("\n✅ PDF RAG is ready!")
    print("Ask questions (type `quit` to exit)\n")

    while True:
        query = input("❓ > ")
        if query.strip().lower() in ("quit", "exit"):
            print("👋 Goodbye!")
            break

        if not query.strip():
            continue

        answer = get_answer(query)
        print("\n✅ Answer:\n", answer)

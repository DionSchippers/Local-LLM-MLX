from mlx_lm import load, generate

PDF_PATH = "./datasets/nederland.pdf"
MODEL_NAME = "mlx-community/Llama-3.2-3B-Instruct"
MARKER = "<|end|><|start|>assistant<|channel|>final<|message|>"
DEFAULT_TOKENS = 1500
TOKEN_LIMIT = 128000

INTRO_PROMPT = "You are a conversational partner about all kind of different things. I want you to have a nice and friendly tone and answer in a nice and kind manner."
RULES_PROMPT = f"""
- Answer in the same language as the query.
- Use the "MEMORY" I sent along to see what was sent by you and by the user.
- If the message starts with "CHAT:" you send it, if it starts with "USER:" it was the user
- The "QUESTION" is the question the user asks now.
"""

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
    CHAT_HISTORY.pop()


def calculate_token_length(query):
    tokens = TOKENIZER.encode(INTRO_PROMPT) + TOKENIZER.encode(RULES_PROMPT) + TOKENIZER.encode(
        CHAT_HISTORY_TEXT) + TOKENIZER.encode(query)
    print(len(tokens))
    if len(tokens) > TOKEN_LIMIT:
        remove_first_messages()
        convert_chat_history_to_text()
        calculate_token_length(query)


def build_prompt(query):
    return f"""You are a friendly and helpful conversational assistant.
Always answer in the same language as the user.
DO NOT add to the conversation history, end after the answer

----- Start of Conversation History -----
{CHAT_HISTORY_TEXT}
----- End of Conversation History -----

User: {query}
Assistant:"""


def get_answer(query):
    global CHAT_HISTORY
    convert_chat_history_to_text()
    print(CHAT_HISTORY)
    calculate_token_length(query)
    prompt = build_prompt(query)
    print(prompt)
    result = generate(
        MODEL,
        TOKENIZER,
        prompt,
        max_tokens=DEFAULT_TOKENS,
    )
    print(result)
    answer = result.strip()
    CHAT_HISTORY.append(f"""User: {query}, Assistant: {answer}""")
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
        print("\n---------------------------------------------\n")

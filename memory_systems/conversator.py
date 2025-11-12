from mlx_lm import load, generate

PDF_PATH = "./datasets/nederland.pdf"
MODEL_NAME = "mlx-community/gpt-oss-20b-MXFP4-Q8"
MARKER = "<|end|><|start|>assistant<|channel|>final<|message|>"
DEFAULT_TOKENS = 1500
TOKEN_LIMIT = 2000
INTRO_PROMPT = "You are a friendly and helpful conversational assistant. Always answer in the same language as the user."
CHAT_HISTORY = []
CHAT_HISTORY_TEXT = ""
MODEL = None
TOKENIZER = None

# =========================
# Step 1: Initialize LLM model and tokenizer
# =========================
# Load the specified LLM model and tokenizer using the `mlx_lm` library.
# This so we can use all this later, but it isn't directly loaded once main gets run.
# =========================
def initialize_model():
    global TOKENIZER, MODEL
    MODEL, TOKENIZER = load(MODEL_NAME)

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


def calculate_token_length(query):
    tokens = TOKENIZER.encode(INTRO_PROMPT) + TOKENIZER.encode(
        CHAT_HISTORY_TEXT) + TOKENIZER.encode(query)
    if len(tokens) > TOKEN_LIMIT:
        remove_first_messages()
        convert_chat_history_to_text()
        calculate_token_length(query)


# =========================
# Step 3: Build the prompt and get answer
# =========================
# We build the prompt by combining the intro prompt, the conversation history, and the current question.
# It is important to keep a little room with your token limit as the generic text in the prompt is not calculated.
# This unfortunately is very difficult as the chat history can vary in length.
# After building the prompt, we generate an answer using the LLM model.
# When the answer is received, we update the chat history with the new question and answer.
# =========================
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
    answer = result.strip().split(MARKER, 1)[1].strip()
    CHAT_HISTORY.append(f"User: {query}")
    CHAT_HISTORY.append(f"Assistant: {answer}\n")
    return answer

# =========================
# Step 4: Run the conversator
# =========================
# This function runs the main loop of the conversator.
# It initializes the model, then enters a loop where it prompts the user for input,
# gets the answer from the LLM, and prints the answer.
# =========================
def run_conversator():
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
        print("\n✅ Answer:\n", answer , '\n')

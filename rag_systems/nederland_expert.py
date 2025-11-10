import re
import fitz
import faiss
from sentence_transformers import SentenceTransformer
from mlx_lm import load, generate

PDF_PATH = "./datasets/nederland.pdf"
MODEL_NAME = "mlx-community/gpt-oss-20b-MXFP4-Q8"
EMBED_MODEL = "paraphrase-multilingual-mpnet-base-v2"
MARKER = "<|end|><|start|>assistant<|channel|>final<|message|>"
DEFAULT_TOKENS = 1500

pdf_text = ""
chunks = []
embedder = None
index = None
model = None
tokenizer = None


def read_pdf(path: str):
    document = fitz.open(path)
    return "".join(page.get_text() for page in document)


def clean_text(text: str):
    text = re.sub(r"<\|.*?\|>", "", text)
    return text


def chunk_text(text: str, chunk_size: int = 1000):
    return re.split(r"\n\s*\n", text)


def build_pdf_index():
    global pdf_text, chunks, embedder, index

    print("📄 Reading PDF...")
    pdf_text_raw = read_pdf(PDF_PATH)

    print("🧹 Cleaning PDF text...")
    pdf_text = clean_text(pdf_text_raw)

    print("✂️ Chunking...")
    chunks = chunk_text(pdf_text)

    print("🧠 Loading embedding model...")
    embedder = SentenceTransformer(EMBED_MODEL)

    print("📐 Creating vector embeddings...")
    vectors = embedder.encode(chunks, convert_to_numpy=True)

    print("📦 Building FAISS index...")
    dimensions = vectors.shape[1]
    index = faiss.IndexFlatL2(dimensions)
    index.add(vectors)

    print(f"✅ Index ready ({len(chunks)} chunks)")


def retrieve_context(query: str, k: int = 6):
    query_vectors = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vectors, k)
    return [chunks[i] for i in indices[0]]


def build_prompt(query, context_blocks):
    context_str = "\n\n".join(context_blocks)
    return f"""
You are an expert of general dutch knowledge. Use the context below to answer the question at the end.

Rules:
- Answer in the same language as the query.
- Use ONLY the context provided.
- If the answer is not there, reply with: "I am sorry, the context unfortunately doesn't contain this information."
- If the answer is in the context, I want you to include a brief explanation of how you came about your answer. Also include the quote from the context that helped you arrive at your answer.

Context:
{context_str}

Question: {query}

Answer:
"""


def rag_answer(query: str):
    context = retrieve_context(query)
    prompt = build_prompt(query, context)
    result = generate(
        model,
        tokenizer,
        prompt,
        max_tokens=DEFAULT_TOKENS,
    )

    return result.strip()


def run_nederland_expert():
    global model, tokenizer

    print("🔧 Building PDF index...")
    build_pdf_index()

    print("🧠 Loading LLM...")
    model, tokenizer = load(MODEL_NAME)

    print("\n✅ PDF RAG is ready!")
    print("Ask questions (type `quit` to exit)\n")

    while True:
        query = input("❓ > ")
        if query.strip().lower() in ("quit", "exit"):
            print("👋 Goodbye!")
            break

        if not query.strip():
            continue

        answer = rag_answer(query)
        if MARKER in answer:
            print("\n✅ Answer:\n", answer.split(MARKER, 1)[1].strip())
        else:
            print("\n❌ Answer too long, try a different question.")
        print("\n---------------------------------------------\n")

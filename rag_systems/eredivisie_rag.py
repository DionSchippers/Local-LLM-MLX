import re
import pandas
import faiss
from sentence_transformers import SentenceTransformer
from mlx_lm import load, generate

CSV_PATH = "datasets/eredivisie_dataset.csv"
MODEL_NAME = "mlx-community/gpt-oss-20b-MXFP4-Q8"
EMBED_MODEL = "multi-qa-MiniLM-L6-cos-v1"
MARKER = "<|end|><|start|>assistant<|channel|>final<|message|>"
DEFAULT_TOKENS = 1500

converted_csv = None
all_seasons = []
all_teams = []
embedder = None
info_snippets = []
index = None
model = None
tokenizer = None


# =========================
# Step 1: Load dataset
# =========================
# Use panda to load CSV.
# Convert each data row into a rext row snippet.
# Create a list of these snippets.
# Make the list of snippets into embeddings via SentenceTransformer.
# --> This is needed for the faiss indexer. This is a tool so that we can search easily for context as the text is converted into vectors.
# --> Vectors are just a numerical representation of the text. This way when we convert the query to text we can find similar text snippets via vector similarity search.
# Build a FAISS index from these vectors.
# =========================

# Load CSV and create snippets
def row_to_text(row):
    if row["champion"]:
        status = "WON the Eredivisie (CHAMPION)"
    else:
        status = f"finished {row['position']}"
    return (
        f"Season: {row['season']}\n"
        f"Team: {row['team']} {status}\n"
    )


def load_csv():
    global converted_csv, all_seasons, all_teams, embedder, info_snippets, index

    converted_csv = pandas.read_csv(CSV_PATH, delimiter=";")

    info_snippets = converted_csv.apply(row_to_text, axis=1).tolist()

    # Create embeddings and FAISS index for lookup
    embedder = SentenceTransformer(EMBED_MODEL)
    vectors = embedder.encode(info_snippets, convert_to_numpy=True)
    dimensions = vectors.shape[1]
    index = faiss.IndexFlatL2(dimensions)
    index.add(vectors)

    # Create lists for aggregation queries
    all_seasons = sorted(converted_csv["season"].unique())
    all_teams = sorted(converted_csv["team"].unique())


# =========================
# Step 2: Load LLM Model & classifier
# =========================
# First we load the LLM model and tokenizer.
# This is used for both classifying the query type and generating the final answer.
# As we are using a RAG approach, we need to know what context we should retrieve for the query. Or if specific context can be left out.
# We do this via a classification prompt that asks the LLM to classify the query into one of several categories.
# --> We let the LLM do this as it can understand the nuances of the query better than a simple keyword-based approach.
# --> For every classification type we can then define specific handling logic.
# Because the LLM returns reasoning text along with the classification, we use a MARKER to extract the real answer.
# We also ask the LLM to return the amount of years to consider for aggregation queries for the same reason as we ask it to classify instead of calculating it.
# =========================
def query_formatter(query):
    formatter_prompt = f"""
You are a helpful assistant that reformats user questions about the Eredivisie football league to be as clear and specific as possible.
I want the strucure to follow this exact format:
Who ... in the ... season(s).
What position ... finished in the ... season(s).
What did ... in the ... season(s).

Examples: 
If the user asks: "Who won in the 2015/2016 season?",
you reformulate it to: "Who won the Eredivisie in the 2015/2016 season."

If the user asks: "What position did Ajax finish in 2020?",
you reformulate it to: "What position Ajax finished in the Eredivisie in the 2020/2021 season."

If the user asks: "In the 2006 season, who finished 3rd?",
you reformulate it to: "Who finished 3rd in the Eredivisie in the 2006/2007 season."

If the user asks: "Which team has won the most titles in the last 10 years?",
you reformulate it to: "Who has won the most Eredivisie titles in the last 10 seasons."

If the user asks: "What was the average position of FC Twente the last 10 years?",
you reformulate it to: "What was the average position of FC Twente in the Eredivisie in the last 10 seasons."

RUlES:
- IMPORTANT — Always follow the exact structure provided.
- IMPORTANT — Respond english.

User question:
{query}
Reformatted question:
"""
    return generate(model, tokenizer, formatter_prompt, max_tokens=DEFAULT_TOKENS).strip().lower().split(MARKER, 1)[
        1].strip()


# classify the type of query and what to do with it
def classify_query(query):
    classifier_prompt = f"""
You are a classification system.

Classify the user's question into one of the following categories:

1) "lookup" — can be answered directly from a single season fact (e.g. "Who won in the 2015/2016 season?", "What position did Ajax finish in 2020?" , "Who finished 4th in the 2006 season", etc.)
2) "aggregate-winners" — requires combining multiple season results (e.g. "Which team has won the most titles in the last 10 years?", "List all champions from 2000 to 2010?", etc.)
3) "aggregate-team" — requires combining multiple season results for a specific team (e.g. "What was the average position of FC Twente the last 10 years?", "List all results for PSV from 2000 to 2010?", "What team got the most 8th places in the last 15 years", etc.)
4) "aggregate-simple" — requires all results of one single season (e.g. "What was the top 10 of last season", "Who where the bottem 3 in 2005",  "What was the top 10 in 2009?", etc.)
5) "unknown" — unclear

Answer with ONLY one word: lookup, aggregate-winners, aggregate-team or unknown.

NOTE: The sentences can be in different forms and languages, so be sure to understand the question well.


User question:
{query}

Answer:
"""
    result = generate(model, tokenizer, classifier_prompt, max_tokens=DEFAULT_TOKENS).strip().lower()
    category = result

    if MARKER in category:
        category = category.split(MARKER, 1)[1].strip()

    if category in ("lookup", "aggregate-winners", "aggregate-team", "aggregate-simple", "unknown"):
        return category


# Extract number of years for aggregation queries
def find_years_for_aggregations_query(query):
    classifier_prompt = f"""
You are a classification system. We want to know how many years to consider for this users aggregation query, so please extract that information from the question.

Answer with ONLY the last season and the amount of years, here are some examples:
If the question is like: "Which team has won the most titles in the last 12 years?" in this context this question is send at the end of 2025 which means the last full completed season is 2024/2025 so you return: 2024/25, 12
If the question is like: "Which team has won the most titles in the last decade?" in this context this question is send at the beginning of 2022 which means the last full completed season is 2020/2021 so you return: 2020/21, 10
If the question is like: "Which team has won the most titles between the 2000 and 2009 seasons" The question has a specified end season, so you pick that as your last season. In this case that is 2009/2010 as there is not specified the beginning or end of that year. You calculate from the 2000/2001 to the 2009/20010 season, which is you 9 seasons so you return: 2009/10, 9
If the question is like: "What was the top 10 of the 2015 season?" you return: 2015/16, 1

NOTE: The sentences can be in different forms and languages, so be sure to understand the question well.

Context:
The current date is {pandas.Timestamp.now().strftime("%Y-%m-%d")}

User question:
{query}

Answer:
"""
    result = generate(model, tokenizer, classifier_prompt, max_tokens=DEFAULT_TOKENS).strip().lower()
    if MARKER in result:
        extracted_result = result.split(MARKER, 1)[1].strip()
        last_season, seasons = extracted_result.split(", ")
        return last_season, int(seasons)


# =========================
# Step 3: Define the tooling the RAG system can use
# =========================
# As stated in step 2, we have several types of queries.
# Each type requires a different context we can provide to the LLM.
# We do this instead of providing all context for every query, as this would be inefficient and could lead to worse answers.
# Besides this we also use this instead of fine-tuning, as this only learns the LLM to recognize patterns which will not generalize well.
# For lookup queries we provide simple snippets straight from the dataset.
# For aggregation queries we use helper functions to compute the required information from the dataset and then send the structured results back into the LLM for natural language generation.
# Finally, for all other questions we specify a general prompt that pre-specifies it does not have this information but will answer to the best of its ability.
# We also create the prompt builder here as it is the last step where the actual RAG is used
# There is some specific tooling here for this use case as we use numbers here, so we need to find those things directly in the dataset.
# =========================

# Specific helpers
# -------------------
def extract_season_from_query(query):
    match = re.search(r"(19|20)\d{2}", query)
    if not match:
        return None
    year = match.group(0)
    next_year = str(int(year) + 1)[2:]
    return f"{year}/{next_year}"


def extract_position_from_query(query):
    match = re.search(r"(\d+)(st|nd|rd|th)?", query)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def structured_lookup(query):
    season = extract_season_from_query(query)
    position = extract_position_from_query(query)

    if season and position:
        filtered = converted_csv[
            (converted_csv["season"] == season)
            & (converted_csv["position"] == position)
            ]
        if not filtered.empty:
            row = filtered.iloc[0]
            return row_to_text(row)
    return None


# -------------------

# Build context for lookup questions
def lookup_context_builder(query):
    season = extract_season_from_query(query)
    position = extract_position_from_query(query)

    if season:
        season_rows = converted_csv[converted_csv["season"] == season]
        if position:
            season_pos_rows = season_rows[season_rows["position"] == position]
            if not season_pos_rows.empty:
                return [row_to_text(r) for _, r in season_pos_rows.iterrows()]
        if not season_rows.empty:
            return [row_to_text(r) for _, r in season_rows.iterrows()]

    query_vec = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, k=100)
    return [info_snippets[i] for i in indices[0]]


def titles_last_specified_years(last_season, seasons):
    end_index = all_seasons.index(last_season)
    start_index = max(0, end_index - seasons + 1)
    context_seasons = all_seasons[start_index: end_index + 1]
    winners = {}
    for season in context_seasons:
        season_winner = converted_csv.query("season == @season and champion == True")
        if not season_winner.empty:
            team = season_winner.iloc[0]["team"]
            winners[season] = team

    return winners


def statistics_of_all_teams_in_specified_years(last_season, seasons):
    end_index = all_seasons.index(last_season)
    start_index = max(0, end_index - seasons + 1)
    context_seasons = all_seasons[start_index: end_index + 1]
    results = {}
    for season in context_seasons:
        for team in all_teams:
            season_stats = converted_csv.query("season == @season and team == @team")
            if not season_stats.empty:
                results[season, team] = season_stats.iloc[0].to_dict()

    return results


def statistics_specific_season(season):
    season_results = converted_csv.query("season == @season").sort_values(by="position")
    results = []
    for _, row in season_results.iterrows():
        results.append(row.to_dict())
    return results


# Build context for aggregation questions
def aggregate_context_builder(query, aggregation_type):
    last_season, seasons = find_years_for_aggregations_query(query)

    if aggregation_type == "aggregate-winners":
        winners = titles_last_specified_years(last_season, seasons)
        facts = {
            "seasons": seasons,
            "winners": winners
        }
        return [str(facts)]

    if aggregation_type == "aggregate-team":
        results = statistics_of_all_teams_in_specified_years(last_season, seasons)
        facts = {
            "seasons": seasons,
            "results": results
        }
        return [str(facts)]

    if aggregation_type == "aggregate-simple":
        results = statistics_specific_season(last_season)
        facts = {
            "seasons": seasons,
            "results": results
        }
        return [str(facts)]


# =========================
# Step 4: Build prompts for the final answers
# =========================
# As there are different types of queries, we need different prompts for the final answer generation.
# For lookup queries we build a prompt that includes the retrieved context snippets.
# For aggregation queries we build a prompt that includes the structured facts computed by the helper functions.
# Finally, for unknown queries we build a generic prompt that tells the LLM it does not have the information, but we want it to try and come with an answer anyway.
# This is done to ensure the LLM provides answers that are as relevant as possible to the user's query.
# These prompts are then used in the main QA router to generate the final answer.
# =========================

def build_lookup_prompt(query, context):
    return f"""
You are a football expert. You specifically know a lot about the Eredivisie. Every question is about the Eredivisie.

Use ONLY the following context to answer the question.
If the answer is not in the context, say: "Unfortunately, I don't have that information."

Context:
{context}

Question: {query}

Rules:
- IMPORTANT — Use the info from the context.
- IMPORTANT — Respond in the same language as the query.

Answer:
"""


def build_aggregate_prompt(query, facts):
    return f"""
You are a football expert. You specifically know a lot about the Eredivisie. Every question is about the Eredivisie.

The user asked:
{query}

Here are structured database results (true facts):
{facts}

Create a helpful, natural-language answer based on these facts.
Rules:
- IMPORTANT — Only use information from facts
- IMPORTANT — Respond in the same language as the query
- Mention ties if multiple teams have the same number of titles
- Include seasons considered and seasons won
- If averages are asked, round to the nearest whole number
- There is always one team per position per season

Answer:
"""


def build_generic_prompt(query):
    return f"""
You are a football expert. You specifically know a lot about the Eredivisie. Every question is about the Eredivisie.

The user asked:
{query}

Unfortunately, we don't have the specific information to answer this question directly.

However, based on your expertise, please provide the best possible answer you can.

Rules:
- IMPORTANT — Clearly state that the information is not available and your answer is a guess and not based on data.
- IMPORTANT — Respond in the same language as the query

Answer:
"""


# =========================
# Step 5: Route the query and generate the final answer
# =========================
# The main function that routes the query based on its classification.
# Depending on the classification, it builds the appropriate context and prompt, then generates the final answer using the LLM.
# =========================
def rag_answer(query):
    print("🔍 Reading query...")
    formatted_query = query_formatter(query)

    print("🛠️ Doing research...")
    task = classify_query(formatted_query)
    prompt = ""
    if "lookup" in task:
        context = lookup_context_builder(formatted_query)
        prompt = build_lookup_prompt(query, context)

    if "aggregate" in task:
        context = aggregate_context_builder(formatted_query, task)
        prompt = build_aggregate_prompt(query, context)

    if "unknown" in task:
        prompt = build_generic_prompt(query)

    print("✍️ Generating answer...")
    return generate(model, tokenizer, prompt, max_tokens=DEFAULT_TOKENS)


def run_eredivisie_rag():
    global model, tokenizer, embedder, index, info_snippets

    print("📦 Loading Eredivisie system...")
    load_csv()  # builds CSV + FAISS

    print("📦 Loading LLM...")
    model, tokenizer = load(MODEL_NAME)
    print("Eredivisie RAG is now running!")
    print("Type your question, or type 'quit' to exit.\n")

    while True:
        query = input("❓ > ")

        # Exit condition
        if query.strip().lower() in ["quit", "exit", "q"]:
            print("👋 Goodbye!")
            break

        # Empty input → ignore
        if not query.strip():
            continue

        answer = rag_answer(query)
        if MARKER in answer:
            print("\n✅ Answer:\n", answer.split(MARKER, 1)[1].strip())
        else:
            print("\n❌ Answer too long, try a different question.")

        print("\n-----------------------------------\n")

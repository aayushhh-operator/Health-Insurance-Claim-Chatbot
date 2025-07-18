import os
import re
import pickle
import faiss
import numpy as np
from flask import Flask, render_template, request
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

app = Flask(__name__)

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load LLM (Phi-3)
llm = Llama(
    model_path="path_to_gguf_model", 
    n_ctx=4096,
    n_threads=6,
    temperature=0.2,
    top_k=50,
    top_p=0.95,
    repeat_penalty=1.1,
    chat_format="chatml"
)

# Load or build FAISS index
if os.path.exists("db/faiss_index") and os.path.exists("db/docs.pkl"):
    index = faiss.read_index("db/faiss_index")
    with open("db/docs.pkl", "rb") as f:
        documents = pickle.load(f)
else:
    # Extract text from PDF
    text = extract_text("data/icici_lombard.pdf")

    # Chunk the text
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    embeddings = embed_model.encode(chunks)

    # Create FAISS index
    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    os.makedirs("db", exist_ok=True)
    faiss.write_index(index, "db/faiss_index")
    with open("db/docs.pkl", "wb") as f:
        pickle.dump(chunks, f)

    documents = chunks

@app.route("/", methods=["GET", "POST"])
def home():
    claim_status = ""
    justification = ""

    if request.method == "POST":
        query = request.form["query"]

        # Embed and search
        query_embedding = embed_model.encode([query])
        D, I = index.search(np.array(query_embedding), k=5)
        retrieved = "\n".join([documents[i] for i in I[0]])

        # Chat-style message formatting for Phi-3
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful insurance claim assistant for ICICI Lombard. "
                    "Based on the ICICI health insurance policy document and the user query, "
                    "determine whether the claim is valid or invalid. "
                    "Begin your response with 'Claim Status: Valid' or 'Claim Status: Invalid'. "
                    "Then explain the reason in 150–200 words using relevant policy clauses."
                ),
            },
            {
                "role": "user",
                "content": f"""Policy Document:
{retrieved}

User Query:
{query}""",
            },
        ]

        # Generate model response
        output = llm.create_chat_completion(messages=messages)
        full_response = output["choices"][0]["message"]["content"].strip()

        # Parse response
        status_match = re.search(r"(Claim Status: Valid|Claim Status: Invalid)", full_response, re.IGNORECASE)
        claim_status = status_match.group(1) if status_match else "Claim Status: Unknown"
        justification = full_response.replace(claim_status, "").strip()

    return render_template("index.html", claim_status=claim_status, justification=justification)

if __name__ == "__main__":
    app.run(debug=True)

import re
import streamlit as st
from transformers import pipeline
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from phones_data import PHONES

# =====================================================
# STREAMLIT CONFIG
# =====================================================

st.set_page_config(
    page_title="AI Smartphone Recommendation (RAG)",
    page_icon="📱",
    layout="centered"
)

st.title("📱 AI Smartphone Recommendation (RAG)")
st.write("Ask things like: **best phones under 90000**")

# =====================================================
# LOAD HEAVY RESOURCES (CACHED)
# =====================================================

@st.cache_resource
def load_resources():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    generator = pipeline(
        "text-generation",
        model="google/flan-t5-base",
        max_new_tokens=250
    )
    return embeddings, generator

# =====================================================
# HELPER FUNCTIONS
# =====================================================

def extract_budget(query):
    match = re.search(r"under\s*(\d+)", query)
    return int(match.group(1)) if match else None


def phone_score(phone):
    name, price, desc, link = phone
    score = price / 1000  # higher price = higher quality
    keywords = ["Ultra", "Pro", "Plus", "Flagship", "Pixel", "iPhone", "Galaxy S"]
    for kw in keywords:
        if kw.lower() in name.lower():
            score += 20
    return score


def recommend_phones(query, budget, embeddings, generator):
    # 1️⃣ Budget filter
    eligible = [p for p in PHONES if p[1] <= budget]

    if not eligible:
        return "❌ No phones found in this budget."

    # 2️⃣ Rank by quality
    eligible.sort(key=phone_score, reverse=True)

    # 3️⃣ Keep top-quality phones only
    top_candidates = eligible[:12]

    # 4️⃣ Create documents
    docs = []
    for name, price, desc, link in top_candidates:
        docs.append(
            Document(
                page_content=f"""
Phone: {name}
Price: ₹{price}
Highlights: {desc}
Buy Link: {link}
"""
            )
        )

    # 5️⃣ Semantic retrieval
    vector_db = FAISS.from_documents(docs, embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 4})
    retrieved_docs = retriever.invoke(query)

    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # 6️⃣ Generation
    prompt = f"""

Context:
{context}

User Query:
{query}

Requirements:
- All phones must be under ₹{budget}
- Recommend only TOP quality phones
- Avoid cheap entry-level phones
- Include Buy Links
"""

    output = generator(prompt)
    return output[0]["generated_text"]

# =====================================================
# APP UI
# =====================================================

with st.spinner("🔄 Loading AI resources..."):
    embeddings, generator = load_resources()

st.success("✅ AI resources loaded")

query = st.text_input(
    "Enter your query:",
    placeholder="best phones under 90000"
)

if query:
    budget = extract_budget(query)

    if not budget:
        st.warning("⚠️ Please specify budget like **under 50000**")
    else:
        with st.spinner("🤖 Finding best phones for you..."):
            result = recommend_phones(query, budget, embeddings, generator)

        st.subheader("✅ Recommended Phones")
        st.markdown(result)

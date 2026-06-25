import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("NVIDIA_API_KEY")
if not api_key or api_key == "nvapi-your-key-here":
    print("ERROR: Please set your actual NVIDIA_API_KEY in the .env file.")
    sys.exit(1)

client = OpenAI(
    api_key=api_key,
    base_url="https://integrate.api.nvidia.com/v1"
)


def build_prompt(user_query, retrieved_faqs):
    context = "\n\n".join(
        [f"Q: {f['question']}\nA: {f['answer']}" for f in retrieved_faqs]
    )

    system_prompt = (
        "You are a support assistant for PulseDesk, a SaaS helpdesk product. "
        "Answer the user's question using ONLY the FAQ context provided below. "
        "If the context does not contain enough information to answer, say "
        "you don't have that information and suggest contacting support. "
        "Do not make up any information not present in the context."
    )

    user_prompt = f"FAQ Context:\n{context}\n\nUser question: {user_query}"

    return system_prompt, user_prompt


def generate_answer(user_query, retrieved_faqs):
    system_prompt, user_prompt = build_prompt(user_query, retrieved_faqs)

    response = client.chat.completions.create(
        model="meta/llama-3.3-70b-instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,   # lower = more focused/less creative, good for factual FAQ answers
        max_tokens=300
    )

    return response.choices[0].message.content


# Quick manual test using fake retrieved FAQs
fake_retrieved = [
    {"question": "How do I reset my password?",
     "answer": "Go to Settings > Account > Security and click 'Reset Password'. You'll receive a reset link via email."}
]

print("--- Testing with matching FAQ ---")
answer = generate_answer("how can I change my password", fake_retrieved)
print("Bot answer:\n", answer)

print("\n--- Testing grounding (unrelated query with empty context) ---")
unrelated_answer = generate_answer("do you sell coffee makers", [])
print("Bot answer:\n", unrelated_answer)

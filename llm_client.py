import os
from typing import Dict, List
from openai import OpenAI

def generate_response(openai_key: str, user_message: str, context: str, 
                     conversation_history: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    """Generate response using OpenAI with context"""

    api_key = openai_key or os.environ.get("OPENAI_API_KEY")

    # TODO: Define system prompt
    system_prompt = """You are an expert NASA mission specialist with deep knowledge of historic 
space missions including Apollo 11, Apollo 13, and the Challenger disaster (STS-51L).
Answer questions based ONLY on the provided context from official NASA documents.
Always cite the source when referencing information.
If the context does not contain enough information, clearly say so.
Do not speculate or add information not present in the context."""

    # TODO: Set context in messages
    messages = [{"role": "system", "content": system_prompt}]

    if context:
        messages.append({
            "role": "user",
            "content": f"Here is the relevant context from NASA documents:\n\n{context}\n\nPlease use this to answer my questions."
        })
        messages.append({
            "role": "assistant",
            "content": "Understood. I have reviewed the NASA documents and will answer based on this context."
        })

    # TODO: Add chat history
    for msg in conversation_history:
        if msg.get("role") in ["user", "assistant"]:
            messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": user_message})

    # TODO: Create OpenAI Client
    client = OpenAI(api_key=api_key, base_url="https://openai.vocareum.com/v1")

    # TODO: Send request to OpenAI
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=1000
    )

    # TODO: Return response
    return response.choices[0].message.content
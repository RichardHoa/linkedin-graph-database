import ollama
import json
from flask import Flask, request, jsonify, render_template
from graph_rag import GraphRAGPipeline
import time

app = Flask(__name__)
pipeline = GraphRAGPipeline()

chat_histories = {}


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    t0 = time.time()
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    session_id = request.remote_addr 

    if session_id not in chat_histories:
        chat_histories[session_id] = []

    history_str = "\n".join([f"{m['role']}: {m['content']}" for m in chat_histories[session_id]])

    router_prompt = f"""
    You are the Lead Orchestrator for a LinkedIn GraphRAG system. 
    Analyze the user's latest question against the conversation history to determine the necessary action and formulate a precise, standalone query for the database.

    SCHEMA CONTEXT:
    {pipeline.cached_context}

    HISTORY:
    {history_str}

    USER LATEST QUESTION: "{user_message}"

    DECISION RULES:
    1. "DIRECT_ANSWER": Use if the question is general chitchat, a greeting, or can be fully and accurately answered relying solely on the provided HISTORY.
    2. "QUERY_GRAPH": Use if the question requires querying the Neo4j database for LinkedIn data, statistics, or professional details not currently in the HISTORY.
    3. "CLARIFY": Use if the question is entirely ambiguous or completely outside the domain of professional/LinkedIn graph data.

    QUERY REFINEMENT RULES (CRITICAL):
    If the action is "QUERY_GRAPH", you MUST generate a `refined_query` that is a fully self-contained, descriptive sentence. The downstream database agent does not have access to the conversation history.
    - Resolve Coreferences: Replace terms like "they", "these", "those", "he", "she", or "it" with the actual entities mentioned in the HISTORY. (e.g., "how many of these devs are junior?" -> "How many junior software developers are there?").
    - Inherit Context: If the user asks a follow-up question (e.g., History: "Who works at Apple?", User: "What about Google?"), combine them into a complete thought (e.g., "Who works at Google?").
    - Be Specific: Include the target professions, companies, or metrics explicitly in the refined string.

    OUTPUT FORMAT: Return ONLY a valid JSON object. Do not include markdown formatting like ```json.
    {{
        "action": "DIRECT_ANSWER" | "QUERY_GRAPH" | "CLARIFY",
        "reply": "Draft the response here for DIRECT_ANSWER or CLARIFY. Leave empty for QUERY_GRAPH.",
        "refined_query": "The fully resolved, self-contained human query (no neo4j code) requiring no prior context."
    }}
    """

    
    router_res = ollama.generate(model="qwen2.5:7b", prompt=router_prompt, format="json")
    router_data = json.loads(router_res['response'])

    if router_data['action'] != "QUERY_GRAPH":
        ai_reply = router_data['reply']
        chat_histories[session_id].append({"role": "user", "content": user_message})
        chat_histories[session_id].append({"role": "assistant", "content": ai_reply})
        return jsonify({"reply": ai_reply, "isErr": False})

    # Proceed to GraphRAG with the refined query
    rag_response = pipeline.run(router_data['refined_query'])
    
    # Check for pipeline errors
    if "error" in rag_response:
        return jsonify({
            "reply": f"System Error: {rag_response['error']}", 
            "isErr": True
        }), 500

    prompt = f"""
    You are a professional assistant analyzing LinkedIn graph data.
    
    CONVERSATION HISTORY:
    {history_str}
    
    USER QUESTION: {rag_response['user_query']}
    RETRIEVED DATA: {json.dumps(rag_response['final_data'], indent=2)}
    
    INSTRUCTIONS:
    - Answer based ONLY on the retrieved data.
    - Be concise and professional.
    """
    
    response = ollama.generate(model="qwen2.5:7b", prompt=prompt)
    ai_reply = response['response']

    chat_histories[session_id].append({"role": "user", "content": user_message})
    chat_histories[session_id].append({"role": "assistant", "content": ai_reply})
    t1 = time.time()

    print(f"{t1-t0:.2f}s for the whole request")

    return jsonify({"reply": ai_reply, "isErr": False})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
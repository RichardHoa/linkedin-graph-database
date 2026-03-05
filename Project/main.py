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
    Role: Lead Orchestrator for a LinkedIn GraphRAG system.
    Task: Determine the required action and resolve conversation context into a standalone query.

    ### CONTEXT
    - SCHEMA: {pipeline.cached_context}
    - HISTORY: {history_str}
    - CURRENT QUESTION: "{user_message}"

    ### DECISION RULES
    1. DIRECT_ANSWER: Greetings, general chitchat, or if the HISTORY already contains the complete answer.
    2. QUERY_GRAPH: Questions requiring LinkedIn-specific stats, professional details, or relationships not in HISTORY.
    3. CLARIFY: Questions that are ambiguous or outside the scope of professional LinkedIn data.

    ### QUERY REFINEMENT INSTRUCTIONS (CRITICAL)
    If action is "QUERY_GRAPH", generate a `refined_query` that is a fully self-contained, descriptive sentence.
    - Contextual Resolution: Resolve all pronouns (e.g., "them", "those", "these") using entities from HISTORY.
    - Example Logic: 
        - History: User asks "how many developers?", AI answers "30,000".
        - Current: User asks "how many of them are junior or fresher?"
        - Refined Query: "Get the total count of developers in junior or fresher positions."

    ### MANDATORY OUTPUT RULES
    - If action is DIRECT_ANSWER or CLARIFY: The "reply" field MUST contain a professional, helpful response. It CANNOT be empty.
    - If action is QUERY_GRAPH: The "reply" field MUST be an empty string (""), and "refined_query" MUST be a standalone sentence resolving all context.

    ### OUTPUT SPECIFICATION
    Return ONLY a JSON object. No markdown formatting.
    {{
        "action": "DIRECT_ANSWER" | "QUERY_GRAPH" | "CLARIFY",
        "reply": "Drafted response for non-graph actions. MUST NOT BE EMPTY for DIRECT_ANSWER or CLARIFY.",
        "refined_query": "Standalone human-language query resolving all historical context."
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
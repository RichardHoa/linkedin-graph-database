import json
from flask import Flask, request, jsonify, render_template
from graph_rag import GraphRAGPipeline
import time
import re

app = Flask(__name__)
pipeline = GraphRAGPipeline()

chat_histories = {}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    t0 = time.time()
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    session_id = request.remote_addr
    if session_id not in chat_histories:
        chat_histories[session_id] = []

    history_str = "\n".join(
        [f"{m['role']}: {m['content']}" for m in chat_histories[session_id]]
    )

    router_prompt = f"""
    Role: Lead Orchestrator for a LinkedIn GraphRAG system.
    Task: Determine the required action and resolve conversation context into a standalone query.

    ### CONTEXT
    - SCHEMA: {pipeline.cached_context}
    - HISTORY: {history_str}
    - CURRENT QUESTION: "{user_message}"

    ### MANDATORY OUTPUT SPECIFICATION
    Return ONLY a JSON object. No markdown.
    {{
        "action": "DIRECT_ANSWER" | "QUERY_GRAPH" | "CLARIFY",
        "reply": "Drafted response for non-graph actions.",
        "refined_query": "Standalone human-language query resolving history."
    }}
    """

    messages = [
        {
            "role": "system",
            "content": "You are a professional assistant. You must respond in valid JSON format only.",
        },
        {"role": "user", "content": router_prompt},
    ]

    # Use HF model instead of Ollama for routing
    router_res = pipeline.generate_completion(messages)
    
    def extract_json(text):
        try:
            # Look for JSON structure { ... }
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1:
                return json.loads(text[start:end+1])
        except Exception as e:
            print(f"JSON Parsing Error: {e}")
        return None

    router_data = extract_json(router_res)
    
    if not router_data:
        print(f"Failed to parse router response: {router_res}")
        router_data = {
            "action": "QUERY_GRAPH",
            "reply": "",
            "refined_query": user_message,
        }
    
    # Defensive key handling to prevent KeyError even if JSON is valid but incomplete
    router_data.setdefault("action", "QUERY_GRAPH")
    router_data.setdefault("reply", "")
    router_data.setdefault("refined_query", user_message)

    if router_data["action"] != "QUERY_GRAPH":
        ai_reply = router_data["reply"]
        chat_histories[session_id].append({"role": "user", "content": user_message})
        chat_histories[session_id].append({"role": "assistant", "content": ai_reply})
        return jsonify({"reply": ai_reply, "isErr": False})

    # Proceed to GraphRAG with the refined query
    rag_response = pipeline.run(router_data["refined_query"])

    # Correct error checking (look into final_data list returned by execute_query)
    final_data = rag_response.get("final_data", [])
    if isinstance(final_data, list) and len(final_data) > 0 and isinstance(final_data[0], dict) and "error" in final_data[0]:
        return (
            jsonify({"reply": f"Database Error: {final_data[0]['error']}", "isErr": True}),
            500,
        )

    answer_prompt = f"""
    You are a professional assistant analyzing LinkedIn graph data.
    CONVERSATION HISTORY: {history_str}
    USER QUESTION: {rag_response['user_query']}
    RETRIEVED DATA: {json.dumps(rag_response['final_data'], indent=2)}
    
    INSTRUCTIONS: Answer based ONLY on the retrieved data. Be concise.
    """

    answer_messages = [
        {"role": "system", "content": "You are a helpful data analyst assistant."},
        {"role": "user", "content": answer_prompt},
    ]

    # Use HF model instead of Ollama for final answer
    ai_reply = pipeline.generate_completion(answer_messages)

    chat_histories[session_id].append({"role": "user", "content": user_message})
    chat_histories[session_id].append({"role": "assistant", "content": ai_reply})

    t1 = time.time()
    print(f"{t1-t0:.2f}s for the request")

    return jsonify({"reply": ai_reply, "isErr": False})


if __name__ == "__main__":
    app.run(debug=True, port=4500)

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
    session_id = request.json.get("session_id", "default_session")
    
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    if session_id not in chat_histories:
        chat_histories[session_id] = []
        
    history = chat_histories[session_id]

    rag_response = pipeline.run(user_message, history=history)
    
    # Save to history
    chat_histories[session_id].append({"role": "user", "content": user_message})
    
    final_data = rag_response.get("final_data", [])
    chat_reply = rag_response.get("chat_reply", "")
    cypher_query = rag_response.get("cypher_query", "N/A")
    
    chat_histories[session_id].append({"role": "assistant", "content": chat_reply})
    
    is_error = False
    
    if isinstance(final_data, list) and len(final_data) > 0 and isinstance(final_data[0], dict) and "error" in final_data[0]:
        is_error = True
        
    reply = chat_reply

    t1 = time.time()
    print(f"{t1-t0:.2f}s for the request")

    return jsonify({"reply": reply, "isErr": is_error})


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=4500)
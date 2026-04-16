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

    rag_response = pipeline.run(user_message)
    
    final_data = rag_response.get("final_data", [])
    cypher_query = rag_response.get("cypher_query", "N/A")
    
    is_error = False
    
    if isinstance(final_data, list) and len(final_data) > 0 and isinstance(final_data[0], dict) and "error" in final_data[0]:
        reply = f"**Database Error:**\n{final_data[0]['error']}\n\n**Generated Cypher:**\n```cypher\n{cypher_query}\n```"
        is_error = True
    elif not final_data:
        reply = f"No results found for the query.\n\n**Generated Cypher:**\n```cypher\n{cypher_query}\n```"
    else:
        # Format output as a markdown code block to display pleasantly back to the user
        formatted_json = json.dumps(final_data, indent=2)
        reply = f"**Results:**\n```json\n{formatted_json}\n```\n\n**Generated Cypher:**\n```cypher\n{cypher_query}\n```"

    t1 = time.time()
    print(f"{t1-t0:.2f}s for the request")

    return jsonify({"reply": reply, "isErr": is_error})


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=4500)
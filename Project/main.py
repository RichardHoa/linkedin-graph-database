import ollama
import json
from flask import Flask, request, jsonify, render_template
from graph_rag import GraphRAGPipeline

app = Flask(__name__)
pipeline = GraphRAGPipeline()

# In-memory history storage
chat_histories = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    session_id = request.remote_addr 
    
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    rag_response = pipeline.run(user_message)
    
    # 2. Handle errors or out-of-scope
    if "error" in rag_response:
        return jsonify({"reply": f"I'm sorry, I couldn't process that: {rag_response['error']}"})

    # 3. Prepare History Context
    if session_id not in chat_histories:
        chat_histories[session_id] = []
    
    history_str = "\n".join([f"{m['role']}: {m['content']}" for m in chat_histories[session_id]])

    prompt = f"""
    You are a professional assistant analyzing LinkedIn graph data.
    
    CONVERSATION HISTORY:
    {history_str}
    
    USER QUESTION: {rag_response['user_query']}
    RETRIEVED DATA: {json.dumps(rag_response['final_data'], indent=2)}
    
    INSTRUCTIONS:
    - Answer based ONLY on the retrieved data.
    - If the data is empty, say you found no matching records.
    - Be concise and professional.
    """
    
    response = ollama.generate(model="qwen3.5:9b", prompt=prompt)
    ai_reply = response['response']

    # 5. Update History
    chat_histories[session_id].append({"role": "user", "content": user_message})
    chat_histories[session_id].append({"role": "assistant", "content": ai_reply})

    return jsonify({"reply": ai_reply})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
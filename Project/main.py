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

    rag_response = pipeline.run(user_message,history_str)
    
    if rag_response.get("is_validation_hit"):
        ai_reply = rag_response['reply']
        
        chat_histories[session_id].append({"role": "user", "content": user_message})
        chat_histories[session_id].append({"role": "assistant", "content": ai_reply})
        
        return jsonify({
            "reply": ai_reply,
            "isErr": False,
            "status": rag_response['status']
        })

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
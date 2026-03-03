from flask import Flask, request, jsonify, render_template
from neo4j import GraphDatabase
from graph_rag import GraphRAGPipeline

app = Flask(__name__)

pipeline = GraphRAGPipeline()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    ai_reply = pipeline.run(user_message) 

    return jsonify({"reply": ai_reply})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
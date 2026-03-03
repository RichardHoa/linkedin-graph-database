from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

chat_history = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    
    chat_history.append({"role": "user", "content": user_message})

    ai_reply = f"I've received your message. We have {len(chat_history)} messages in our history."

    chat_history.append({"role": "assistant", "content": ai_reply})

    return jsonify({"reply": ai_reply})

if __name__ == '__main__':
    app.run(debug=True)
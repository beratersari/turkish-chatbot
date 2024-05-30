import sys
sys.path.append('../')



from flask import Flask, render_template, request, jsonify
from config import chatbot

app = Flask(__name__)

if chatbot=="mistral7b":
    from mistral7b.chat import mistral_chatbot
elif chatbot=="chatbot_from_scratch":
    from chatbot_from_scratch.predict import predict_answer
elif chatbot == "finetune_gpt2":
    from finetune_gpt2.predict import infer
@app.route("/")
def hello():
    return render_template('chat.html')


@app.route("/ask", methods=['POST'])
def ask():
    message = str(request.form['messageText'])
    bot_response= ""
    if chatbot=="mistral7b":
        bot_response = mistral_chatbot(message)
    elif chatbot=="chatbot_from_scratch":
        bot_response = predict_answer(message)
    elif chatbot == "finetune_gpt2":
        bot_response = infer(message)
    else:
        bot_response = "choose a valid model"

    return jsonify({'status': 'OK', 'answer': bot_response})


if __name__ == "__main__":
    app.run()
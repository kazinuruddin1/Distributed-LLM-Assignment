# app.py
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

model_name = None
model = None
tokenizer = None
conversation_history = []

def load_model(selected_model):
    global model_name, model, tokenizer
    model_name = selected_model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

@app.route('/select_model', methods=['POST'])
def select_model():
    data = request.json
    load_model(data['model'])
    return jsonify({"status": "Model loaded successfully"}), 200

@app.route('/query', methods=['POST'])
def query_model():
    data = request.json
    query = data['query']
    inputs = tokenizer.encode(query, return_tensors="pt")
    conversation_history.append(query)
    outputs = model.generate(inputs, max_length=500)
    response = tokenizer.decode(outputs[0])
    conversation_history.append(response)
    return jsonify({"response": response, "history": conversation_history}), 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

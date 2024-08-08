## Part 1: Python Program for Llama2 and Mistral

### Features

1. **Model Selection**: When the program starts, the user can select either Llama2 or Mistral.
2. **Query Handling**: The user can send a query to the selected model and receive an answer.
3. **Context Maintenance**: The program should maintain the conversation context between the user and the LLM.

### Conditions

1. The entire application should be wrapped in Docker.
2. Provide a README.md file with instructions.

### Steps:

1. **Setup Virtual Environment and Dependencies**: Create a virtual environment and install the necessary packages (`transformers`, `flask`, etc.).
2. **Model Initialization**: Create a function to load the selected model.
3. **Query Handling with Context**: Implement a way to handle and store the conversation context.
4. **Flask API**: Create a simple Flask API to interact with the LLM.
5. **Dockerization**: Create a Dockerfile and docker-compose file for the setup.
6. **README.md**: Provide instructions on how to run the application.

### Sample Code

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



# Dockerfile
FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "app.py"]


# docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "5000:5000"


flask
transformers
torch


# Python LLM Server

## Setup

1. Build and run the Docker container:
```sh
docker-compose up --build


curl -X POST -H "Content-Type: application/json" -d '{"model": "Llama2"}' http://localhost:5000/select_model


curl -X POST -H "Content-Type: application/json" -d '{"query": "Hello, how are you?"}' http://localhost:5000/query
curl http://localhost:3000/conversation/1

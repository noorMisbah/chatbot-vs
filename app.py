from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = Flask(__name__)
CORS(app)  # Allow Android app to call this API

# Load your fine-tuned model and tokenizer
model_path = "./fine-tuned-distilgpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get("message", "")
    
    # Format input for your model (adjust based on your training)
    input_text = f"Q: {user_input}\nA:"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    
    # Generate response
    output = model.generate(
        input_ids,
        max_length=100,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id
    )
    
    bot_response = tokenizer.decode(output[0], skip_special_tokens=True)
    # Extract only the "A: ..." part
    bot_response = bot_response.split("A:")[-1].strip()
    
    return jsonify({"response": bot_response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
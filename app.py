from flask import Flask, request, render_template
from tflite_runtime.interpreter import Interpreter
from transformers import BertTokenizer
import numpy as np

# Initialize app and tokenizer
app = Flask(__name__)
tokenizer = BertTokenizer.from_pretrained("google/mobilebert-uncased")

# Load TFLite model
interpreter = Interpreter(model_path="mobilebert_quant_384_20200602.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Fixed passage (context)
PASSAGE = (
    "Pandas live in the bamboo forests of China and are known for their diet of mostly bamboo. "
    "They are native to central China and are considered a national treasure."
)

def preprocess_input(question, passage, max_len=384):
    encoding = tokenizer.encode_plus(
        question,
        passage,
        max_length=max_len,
        truncation=True,
        padding="max_length",
        return_tensors="np"
    )

    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    token_type_ids = encoding["token_type_ids"]

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    return input_ids, attention_mask, token_type_ids, tokens

def postprocess_output(start_logits, end_logits, tokens):
    max_len = len(tokens)
    best_score = float("-inf")
    best_start, best_end = 0, 0

    for start in range(max_len):
        for end in range(start, min(start + 30, max_len)):
            score = start_logits[start] + end_logits[end]
            if score > best_score and start <= end:
                best_score = score
                best_start, best_end = start, end

    if best_start >= len(tokens) or best_end >= len(tokens):
        return "Could not find a valid answer."

    answer_tokens = tokens[best_start:best_end + 1]
    answer = ""
    for token in answer_tokens:
        if token.startswith("##"):
            answer += token[2:]
        elif answer:
            answer += " " + token
        else:
            answer = token

    return answer.strip()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def get_response():
    question = request.form["msg"]
    input_ids, input_mask, segment_ids, tokens = preprocess_input(question, PASSAGE)

    interpreter.set_tensor(input_details[0]["index"], input_ids)
    interpreter.set_tensor(input_details[1]["index"], input_mask)
    interpreter.set_tensor(input_details[2]["index"], segment_ids)
    interpreter.invoke()

    start_logits = interpreter.get_tensor(output_details[0]["index"])[0]
    end_logits = interpreter.get_tensor(output_details[1]["index"])[0]

    answer = postprocess_output(start_logits, end_logits, tokens)
    return answer
     
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

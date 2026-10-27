
from flask import Flask, request, Response
from model_loader import load_model
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from transformers import TextIteratorStreamer
from threading import Thread 
import json

app= Flask(__name__)
model, tokenizer= load_model()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


@app.route('/generate', methods=['POST'])
def handle_generation():
    data= request.json
    prompt= data.get('prompt', '')
    max_new_tokens= 1000
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    streamer=TextIteratorStreamer(tokenizer, skipPrompt=True, skip_special_tokens=True)

    generation_kwargs= dict(
        input_ids=input_ids,
        streamer=streamer,
        num_return_sequences=1, 
        no_repeat_ngram_size=2,
        temperature=0.2,
        top_p=0.9,
        do_sample=True,
        max_new_tokens= max_new_tokens,
    )   

    thread= Thread(target=model.generate, kwargs=generation_kwargs).start()
    def generate_chunks():
        for new_text in streamer:
            # 1. Remove the JSON wrapping and yield the raw text fragment
            yield new_text
            

    # 2. Change mimetype from 'text/event-stream' to 'text/plain'
    return Response(generate_chunks(), mimetype='text/plain')




if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
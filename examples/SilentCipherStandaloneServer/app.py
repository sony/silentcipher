from flask import Flask, request
import json
import torch
import silentcipher

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

model = silentcipher.get_model(
    model_type='44.1k', 
    ckpt_path='../../Models/44_1_khz/73999_iteration', 
    config_path='../../Models/44_1_khz/73999_iteration/hparams.yaml',
    device=device
)


app = Flask(__name__)

@app.route('/encode', methods=['POST'])
def encode(): 
    response = json.dumps(model.encode(request.json['in_path'], request.json['out_path'], request.json['message']))
    print(response)
    return response

@app.route('/decode', methods=['POST'])
def decode():
    response = json.dumps(model.decode(request.json['path'], request.json['phase_shift_decoding']))
    print(response)
    return response

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8001)

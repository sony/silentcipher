from email import message
from flask import Flask, request
import json
import torch
import silentcipher
import yaml

config = yaml.safe_load(open('config.yaml'))

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

models = {}

if config['enable_44k']:
    models['44k'] = silentcipher.get_model(
        model_type='44.1k', 
        ckpt_path='../../Models/44_1_khz/73999_iteration', 
        config_path='../../Models/44_1_khz/73999_iteration/hparams.yaml',
        device=device
    )
if config['enable_16k']:
    models['16k'] = silentcipher.get_model(
        model_type='16k',
        ckpt_path='../../Models/16_khz/97561_iteration', 
        config_path='../../Models/16_khz/97561_iteration/hparams.yaml',
        device=device
    )


app = Flask(__name__)

@app.route('/encode', methods=['POST'])
def encode():
    if request.json['model_type'] == '44k': # type: ignore
        if not config['enable_44k']:
            return json.dumps({'status': False, 'message': 'Please enable the 44k model in the config file to be able to encode using the 44k model'})
        model = models['44k']
    elif request.json['model_type'] == '16k': # type: ignore
        if not config['enable_16k']:
            return json.dumps({'status': False, 'message': 'Please enable the 16k model in the config file to be able to encode using the 16k model'})
        model = models['16k']
    else:
        return json.dumps({'status': False, 'message': f'{request.json["model_type"]} Model type not implemented'}) # type: ignore
    
    if request.json['message_sdr'] is not None: # type: ignore
        message_sdr = float(request.json['message_sdr']) # type: ignore
    else:
        message_sdr = None
    
    response = json.dumps(model.encode(request.json['in_path'], request.json['out_path'], request.json['message'], message_sdr)) # type: ignore
    return response

@app.route('/decode', methods=['POST'])
def decode():
    if request.json['model_type'] == '44k': # type: ignore
        if not config['enable_44k']:
            return json.dumps({'status': False, 'message': 'Please enable the 44k model in the config file to be able to encode using the 44k model'})
        model = models['44k']
    elif request.json['model_type'] == '16k': # type: ignore
        if not config['enable_16k']:
            return json.dumps({'status': False, 'message': 'Please enable the 16k model in the config file to be able to encode using the 16k model'})
        model = models['16k']
    else:
        return json.dumps({'status': False, 'message': f'{request.json["model_type"]} Model type not implemented'}) # type: ignore
    
    response = json.dumps(model.decode(request.json['path'], request.json['phase_shift_decoding'])) # type: ignore
    return response

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8001)

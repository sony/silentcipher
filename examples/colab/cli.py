import silentcipher
import librosa
import json
import argparse


def encode(model, filename, encoded_filename, msg=None, message_sdr=None, calc_sdr=True, disable_checks=False):
    try:
        return model.encode(filename, encoded_filename, msg, message_sdr=message_sdr, calc_sdr=calc_sdr, disable_checks=disable_checks)
    except:
        return {'status': False}

def decode(model, encoded_filename, phase_shift_decoding=True):
    try:
        result = model.decode(encoded_filename, phase_shift_decoding)
    except:
        result = {'status': False}
    return result


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    
    parser.add_argument('--filename', type=str, help='Input audio file', required=True)
    parser.add_argument('--results_json_path', type=str, help='Store the results of the run', required=True)
    parser.add_argument('--mode', type=str, help='Mode of operation', choices=['encode', 'decode'], required=True)
    parser.add_argument('--model_type', type=str, help='44.1khz or 16khz', choices=['44.1k', '16k'], required=True)
    parser.add_argument('--use_gpu', type=bool, help='Whether to use cuda or not', default=True)

    # Encode parameters
    parser.add_argument('--msg', type=str, help='Message to encode in the form of five 8-bit comma separated digits, eg. 111,222,123,234,12', default=None)
    parser.add_argument('--encoded_filename', type=str, help='Path to save the encoded audio file', default=None)
    parser.add_argument('--message_sdr', type=str, help='Message SDR to encode | default is None which will use the default value of SDR of the model', default=None)

    # Decode parameters
    parser.add_argument('--phase_shift_decoding', type=bool, help='Whether to detect messages even in the case when the audio may be cropped. WARNING! INCREASES THE DETECTION TIME CONSIDERABLY', default=True)
    args = parser.parse_args()

    if args.model_type == '44.1k':
        model = silentcipher.get_model(model_type='44.1k', device='cuda' if args.use_gpu else 'cpu')
    elif args.model_type == '16k':
        model = silentcipher.get_model(model_type='16k', device='cuda' if args.use_gpu else 'cpu')
    else:
        raise ValueError('Invalid model type')
    
    if args.mode == 'encode':
        assert args.filename is not None, 'Path to original file is required for encoding'
        assert args.encoded_filename is not None, 'Path to save the encoded file is required'
        assert args.msg is not None, 'Message to encode is required'

        msg = [int(i) for i in args.msg.split(',')]

        with open(args.results_json_path, 'w') as f:
            result = encode(model, args.filename, args.encoded_filename, msg, args.message_sdr)
            json.dump(result, f, indent=4)
    elif args.mode == 'decode':
        assert args.filename is not None, 'Path to encoded file is required for decoding'
        with open(args.results_json_path, 'w') as f:
            result = decode(model, args.filename, args.phase_shift_decoding)
            json.dump(result, f, indent=4)

    # Encoding Example:

    # python cli.py --filename test.wav --mode encode --model_type 44.1k --use_gpu true --msg 12,13,14,15,16 --encoded_filename enc_cli.wav --results_json_path out.json

    # Decoding Example:

    # python cli.py --filename enc_cli.wav --mode decode --model_type 44.1k --use_gpu true --results_json_path decode_out.json
    
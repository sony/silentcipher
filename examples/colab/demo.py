import silentcipher
import librosa

device='cuda'

model = silentcipher.get_model(
    model_type='44.1k',
    ckpt_path='../../Models/44_1_khz/73999_iteration', 
    config_path='../../Models/44_1_khz/73999_iteration/hparams.yaml',
    device=device
)

"""
For 16khz model

model = silentcipher.get_model(
    model_type='16k',
    ckpt_path='../Models/16_khz/97561_iteration', 
    config_path='../Models/16_khz/97561_iteration/hparams.yaml',
    device=device
)
"""

def test(y, sr, filename, encoded_filename):
    
    # The message should be in the form of five 8-bit characters, giving a total message capacity of 40 bits

    # Single Channel encoding example
    # If you want to really speedup the watermarking process then set disable_checks=True, but beware as this may cause unexpected results
    # Set calc_sdr=False to not calculate the SDR of the encoded audio
    encoded, sdr = model.encode_wav(y, sr, [123, 234, 111, 222, 11], message_sdr=None, calc_sdr=True, disable_checks=False)

    # You can specify the message SDR (in dB) as a float along with the encode_wav function. But this may result in unexpected detection accuracy
    # encoded, sdr = model.encode_wav(y, sr, [123, 234, 111, 222, 11], message_sdr=47)

    # For multi-channel audio, you can use the following function
    # Here y is a 2 channel audio with shape [num_frames, num_channels] and you can specify the message for each channel
    # encoded, sdr = model.encode_wav(y, sr, [[123, 234, 111, 222, 11], [132, 214, 121, 122, 211]])

    # Single Channel decoding example
    result = model.decode_wav(encoded, sr, phase_shift_decoding=False)

    if type(result) is list:
        for result_i in result:
            assert result_i['status']
            assert result_i['messages'][0] == [123, 234, 111, 222, 11], result_i['messages'][0]
            assert result_i['confidences'][0] == 0.9746031761169434, result_i['confidences'][0]
    else:
        assert result['status']
        assert result['messages'][0] == [123, 234, 111, 222, 11], result['messages'][0]
        assert result['confidences'][0] == 0.9746031761169434, result['confidences'][0]

    # When decoding multi-channel audio, The result would be a list of dictionaries with the status, message and confidence for each channel

    # Encode from filename

    model.encode(filename, encoded_filename, [123, 234, 111, 222, 11], message_sdr=None, calc_sdr=True, disable_checks=False)
    result = model.decode(encoded_filename, phase_shift_decoding=False)

    if type(result) is list:
        for result_i in result:
            assert result_i['status']
            assert result_i['messages'][0] == [123, 234, 111, 222, 11], result_i['messages'][0]
            assert result_i['confidences'][0] == 0.9746031761169434, result_i['confidences'][0]
    else:
        assert result['status']
        assert result['messages'][0] == [123, 234, 111, 222, 11], result['messages'][0]
        assert result['confidences'][0] == 0.9746031761169434, result['confidences'][0]

    # When decoding multi-channel audio, The result would be a list of dictionaries with the status, message and confidence for each channel


y, sr = librosa.load('test.wav', sr=None)
test(y, sr, 'test.wav', 'encoded_test.wav')
y, sr = librosa.load('test_multichannel.wav', sr=None)
test(y, sr, 'test_multichannel.wav', 'encoded_test_multichannel.wav')

print('Test Successful')
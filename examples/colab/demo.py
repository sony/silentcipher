import silentcipher
import librosa

model = silentcipher.get_model(
    model_type='44.1k',
    ckpt_path='../../Models/44_1_khz/73999_iteration', 
    config_path='../../Models/44_1_khz/73999_iteration/hparams.yaml'
)

"""
For 16khz model

model = silentcipher.get_model(
    model_type='16k',
    ckpt_path='../Models/16_khz/97561_iteration', 
    config_path='../Models/16_khz/97561_iteration/hparams.yaml'
)
"""

# Encode from waveform

y, sr = librosa.load('test.wav', sr=None)

# The message should be in the form of five 8-bit characters, giving a total message capacity of 40 bits 

encoded, sdr = model.encode_wav(y, sr, [123, 234, 111, 222, 11])

# You can specify the message SDR (in dB) along with the encode_wav function. But this may result in unexpected detection accuracy
# encoded, sdr = model.encode_wav(y, sr, [123, 234, 111, 222, 11], message_sdr=47)

result = model.decode_wav(encoded, sr, phase_shift_decoding=False)

assert result['status']
assert result['messages'][0] == [123, 234, 111, 222, 11], result['messages'][0]
assert result['confidences'][0] == 1, result['confidences'][0]

# Encode from filename

# The message should be in the form of five 8-bit characters, giving a total message capacity of 40 bits 

model.encode('test.wav', 'encoded.wav', [123, 234, 111, 222, 11])

# You can specify the message SDR (in dB) along with the encode function. But this may result in unexpected detection accuracy
# model.encode('test.wav', 'encoded.wav', [123, 234, 111, 222, 11], message_sdr=47)

result = model.decode('encoded.wav', phase_shift_decoding=False)

assert result['messages'][0] == [123, 234, 111, 222, 11], result['messages'][0]
assert result['confidences'][0] == 1, result['confidences'][0]


print('Test Successful')
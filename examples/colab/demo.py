import silentcipher

model = silentcipher.get_model(
    model_type='44.1k', # 16k
    ckpt_path='../Models/44_1_khz/73999_iteration', 
    config_path='../Models/44_1_khz/73999_iteration/hparams.yaml'
)

# Encode from waveform

y, sr = librosa.load('test.wav', sr=None)
encoded, sdr = model.encode_wav(y, sr, [123, 234, 111, 222, 11])
result = model.decode_wav(encoded, sr, phase_shift_decoding=False)

assert result['status']
assert result['messages'][0] == [123, 234, 111, 222, 11], result['messages'][0]
assert result['confidences'][0] == 1, result['confidences'][0]

# Encode from filename

model.encode('test.wav', 'encoded.wav', [123, 234, 111, 222, 11])
result = model.decode('encoded.wav', phase_shift_decoding=False)

assert result['messages'][0] == [123, 234, 111, 222, 11], result['messages'][0]
assert result['confidences'][0] == 1, result['confidences'][0]


print('Test Successful')
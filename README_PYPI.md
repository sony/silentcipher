# :loud_sound: SilentCipher: Deep Audio Watermarking: [Link to arxiv](https://arxiv.org/abs/2406.03822)

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.8+-blue?style=for-the-badge&logo=python&logoColor=white"></a>

Code for SilentCipher, a method of embedding watermarking in any audio with state-of-the-art robustness.<br>
Currently this repository supports audio at 16kHz and 44.1kHz.<br>
Checkout our [paper](https://arxiv.org/abs/2406.03822) for more details.

[[`arXiv`](https://arxiv.org/abs/2406.03822)]
[[`Colab notebook`](https://colab.research.google.com/github/sony/silentcipher/blob/master/examples/colab/demo.ipynb)]
<!-- [[🤗`Hugging Face`](HUGGINGFACE)] -->

# Summary

In this paper, we address artefacts introduces by Deep learning-based watermarking methods and introduce a way to remove the need for perceptual losses which leads to stable training allowing us to achieve SOTA in terms of both perceptual quality and robustness against distortion. Unlike previous methods which work on 16kHz sampling rate, we also showcase our results on 44.1kHz sampling rates opening the path for practical applications.

# Abstract

In the realm of audio watermarking, it is challenging to simultaneously encode imperceptible messages while enhancing the message capacity and robustness. Although recent advancements in deep learning-based methods bolster the message capacity and robustness over traditional methods, the encoded messages introduce audible artefacts that restricts their usage in professional settings. In this study, we introduce three key innovations. Firstly, our work is the first deep learning-based model to integrate psychoacoustic model based thresholding to achieve imperceptible watermarks. Secondly, we introduce psuedo-differentiable compression layers, enhancing the robustness of our watermarking algorithm. Lastly, we introduce a method to eliminate the need for perceptual losses, enabling us to achieve SOTA in both robustness as well as imperceptible watermarking. Our contributions lead us to SilentCipher, a model enabling users to encode messages within audio signals sampled at 44.1kHz.

# :mate: Installation

SilentCipher requires Python >=3.8.<br>
I would recommend using a python virtual environment.
```
python -m venv env
source env/bin/activate
```

To install from PyPI:

```
pip install silentcipher
```
To install from source: Clone this repo and install in editable mode:
```
git clone [GIT_URL]
pip install build
python -m build
pip install dist/<package>.whl
```

# :gear: Models

**Note**: We will be uploading the model checkpoints soon. Stay Tuned!<br>
**Note**: We are working to release the training code for anyone wants to build their own watermarker. Stay tuned !


<!-- You can find all the model checkpoints on the [Hugging Face Hub](HUGGINGFACE). We provide the checkpoints for the following models:

- [SilentCipher-44.1khz](It takes a 44.1kHz audio signal as input along with the message to be embedded and generates a watermarked audio of the same size as output. This model may be useful for professional applications).
- [SilentCipher-16khz](It takes a 16kHz audio signal as input along with the message to be embedded and generates a watermarked audio of the same size as output. This model may be useful for deep learning based applications which generate audio inherently at 16kHz). -->


# :abacus: Usage

SilentCipher provides a simple API to watermark and detect the watermarks from an audio sample. Example usage:

```python
import librosa
import silentcipher

model = silentcipher.get_model(
    model_type='44.1k', # 16k
    ckpt_path='../Models/44_1_khz/73999_iteration', 
    config_path='../Models/44_1_khz/73999_iteration/hparams.yaml'
)

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
```

# Want to contribute?

 We welcome Pull Requests with improvements or suggestions.
 If you want to flag an issue or propose an improvement, but dont' know how to realize it, create a GitHub Issue.

<!-- # Troubleshooting -->
# License

- The code in this repository is released under the license as found in the [LICENSE file](LICENSE).

# Maintainers:
- [Mayank Kumar Singh](https://github.com/mayank-git-hub)

# Citation

If you find this repository useful, please consider giving a star :star: and please cite as:

```
@inproceedings{singh24_interspeech,
  author={Mayank Kumar Singh and Naoya Takahashi and Weihsiang Liao and Yuki Mitsufuji},
  title={{SilentCipher: Deep Audio Watermarking}},
  year=2024,
  booktitle={Proc. INTERSPEECH 2024},
}
```

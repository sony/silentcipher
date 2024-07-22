# :loud_sound: SilentCipher: Deep Audio Watermarking: [Link to arxiv](https://arxiv.org/abs/2406.03822)

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.8+-blue?style=for-the-badge&logo=python&logoColor=white"></a>

Code for SilentCipher, a method of embedding watermarking in any audio with state-of-the-art robustness.<br>
Currently this repository supports audio at 16kHz and 44.1kHz.<br>
Checkout our [paper](https://arxiv.org/abs/2406.03822) for more details.<br>
<br>
We have posted some examples from existing watermarking algorithms and how they compare to our watermarking method at [EXAMPLES](https://interspeech2024.github.io/silentcipher/) 

[[`arXiv`](https://arxiv.org/abs/2406.03822)]
[[`Colab notebook`](https://colab.research.google.com/github/sony/silentcipher/blob/master/examples/colab/demo.ipynb)]
[[🤗`Hugging Face`](https://huggingface.co/Sony/SilentCipher)]

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
To install from source: Clone this repo and run the following commands:
```
git clone https://github.com/sony/silentcipher.git
pip install build
python -m build
pip install dist/<package>.whl
```

# :gear: Models

Find the latest models for 44.1kHz and 16kHz sampling rate in the release section of this repository [RELEASE](https://github.com/sony/silentcipher/releases)<br>
The models have also been released on [HuggingFace](https://huggingface.co/Sony/SilentCipher)<br>

**Note**: We are working to release the training code for anyone wants to build their own watermarker. Stay tuned !

# :abacus: Usage

SilentCipher provides a simple API to watermark and detect the watermarks from an audio sample.<br>
<br>
We showcase it in multiple ways as shown in the examples directory.<br>
We provide a simple flask server as documented in [README_FLASK](https://github.com/sony/silentcipher/tree/master/examples/SilentCipherStandaloneServer)<br>
You can also find a simple front-end and backend server which can be used to demonstrate the applications of silentcipher [README_UI](https://github.com/sony/silentcipher/tree/master/examples/WaterMarkingWebsite)<br>
Some simple demo examples are also provided in the [COLAB DIR](https://github.com/sony/silentcipher/tree/master/examples/colab)

Over here we provide an usage in python:

```python
import librosa
import silentcipher

model = silentcipher.get_model(
    model_type='44.1k', # 16k
    device='cuda'  # use 'cpu' if you want to run it without GPUs
)
# By default the model is loaded using hugging face APIs, but you can specify the ckpt_path and config_path manually as well
# ckpt_path='Models/44_1_khz/73999_iteration', 
# config_path='Models/44_1_khz/73999_iteration/hparams.yaml',

# Encode from waveform

y, sr = librosa.load('examples/colab/test.wav', sr=None)

# The message should be in the form of five 8-bit characters, giving a total message capacity of 40 bits 

encoded, sdr = model.encode_wav(y, sr, [123, 234, 111, 222, 11])

# You can specify the message SDR (in dB) along with the encode_wav function. But this may result in unexpected detection accuracy
# encoded, sdr = model.encode_wav(y, sr, [123, 234, 111, 222, 11], message_sdr=47)

# You should set phase_shift_decoding to True when you want the decoder to be robust to audio crops.
# !Warning, this can increase the decode time quite drastically.

result = model.decode_wav(encoded, sr, phase_shift_decoding=False)

print(result['status'])
print(result['messages'][0] == [123, 234, 111, 222, 11])
print(result['confidences'][0])

# Encode from filename

# The message should be in the form of five 8-bit characters, giving a total message capacity of 40 bits 

model.encode('examples/colab/test.wav', 'examples/colab/encoded.wav', [123, 234, 111, 222, 11])

# You can specify the message SDR (in dB) along with the encode function. But this may result in unexpected detection accuracy
# model.encode('test.wav', 'encoded.wav', [123, 234, 111, 222, 11], message_sdr=47)

# You should set phase_shift_decoding to True when you want the decoder to be robust to audio crops.
# !Warning, this can increase the decode time quite drastically.

result = model.decode('examples/colab/encoded.wav', phase_shift_decoding=False)

print(result['messages'][0] == [123, 234, 111, 222, 11], result['messages'][0])
print(result['confidences'][0])
```

# Demo Programs 

1. [Python demo program with more detailed usage](https://github.com/sony/silentcipher/blob/master/examples/colab/demo.py)
2. [Colab Google](https://colab.research.google.com/github/sony/silentcipher/blob/master/examples/colab/demo.ipynb)
3. [A standalone flask server](https://github.com/sony/silentcipher/tree/master/examples/SilentCipherStandaloneServer)
4. [A demo project management UI based on angular + django + flask](https://github.com/sony/silentcipher/tree/master/examples/WaterMarkingWebsite)

# Want to contribute?

 We welcome Pull Requests with improvements or suggestions.
 If you want to flag an issue or propose an improvement, but dont' know how to realize it, create a GitHub Issue.

<!-- # Troubleshooting -->
# License

- The code in this repository is released under the MIT license as found in the [LICENSE file](LICENSE).

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

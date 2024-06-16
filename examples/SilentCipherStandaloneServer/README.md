# Flask App

This is a Flask application that serves as the server for the SilentCipher Standalone Package.

## Installation

1. Follow the installation instructions at [README.md](https://github.com/sony/silentcipher/blob/master/README.md)

2. Navigate to the `SilentCipherStandaloneServer` directory:

    ```bash
    cd SilentCipherStandalonePackage/examples/SilentCipherStandaloneServer
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Start the Flask server:

    ```bash
    CUDA_VISIBLE_DEVICES=0 flask run --host 0.0.0.0 --port 8001
    ```

2. There are two APIs, one for encode and the other for decode

    ```bash

    curl --location 'http://127.0.0.1:8001/encode' \
    --header 'Content-Type: application/json' \
    --data '{
        "model_type": "44k",
        "message_sdr": null,
        "in_path": "../colab/test.wav",
        "out_path": "encoded_flask.wav",
        "message": [111, 222, 121, 131, 141]
    }'
    ```

    ```bash

    curl --location 'http://127.0.0.1:8001/decode' \
    --header 'Content-Type: application/json' \
    --data '{
        "model_type": "44k",
        "path": "encoded_flask.wav",
        "phase_shift_decoding": false
    }'
    ```

## Contributing

Currently the standalone server supports only single channel audio.<br>
Feel free to submit a PR for adding the multi-channel support to the standalone server

## License

This project is licensed under the [MIT License](LICENSE).

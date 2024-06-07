import os
import argparse
import yaml
import time
import numpy as np
import soundfile as sf
from scipy import stats as st
import librosa
from pydub import AudioSegment
import torch
from torch import nn

from .model import Encoder, CarrierDecoder, MsgDecoder
from .stft import STFT

class Model():
    
    def __init__(self, config, device='cpu'):
         
        self.config = config
        self.device = device
        
        self.n_messages = config.n_messages
        self.model_type = config.model_type
        self.message_dim = config.message_dim
        self.message_len = config.message_len

        # model dimensions
        self.enc_conv_dim     = 16
        self.enc_num_repeat   = 3
        self.dec_c_num_repeat = self.enc_num_repeat
        self.dec_m_conv_dim   = 1
        self.dec_m_num_repeat = 8
        self.encoder_out_dim = 32
        self.dec_c_conv_dim = 32*3
            
        self.enc_c = Encoder(n_layers=self.config.enc_n_layers,
                             message_dim=self.message_dim,
                             out_dim=self.encoder_out_dim,
                             message_band_size=self.config.message_band_size,
                             n_fft=self.config.N_FFT)

        self.dec_c = CarrierDecoder(config=self.config,
                                    conv_dim=self.dec_c_conv_dim,
                                    n_layers=self.config.dec_c_n_layers,
                                    message_band_size=self.config.message_band_size)

        self.dec_m = [MsgDecoder(message_dim=self.message_dim,
                                 message_band_size=self.config.message_band_size) for _ in range(self.n_messages)]
        # ------ make parallel ------
        if self.device != 'cpu':
            self.enc_c = nn.DataParallel(self.enc_c.to(self.device))
            self.dec_c = nn.DataParallel(self.dec_c.to(self.device))
            self.dec_m = [nn.DataParallel(m.to(self.device)) for m in self.dec_m]
        
        self.average_energy_VCTK=0.002837200844477648
        self.stft = STFT(self.config.N_FFT, self.config.HOP_LENGTH)
        self.stft.to(self.device)
        self.load_models(config.load_ckpt)
        self.sr = self.config.SR

    def letters_encoding(self, patch_len, message_lst):
         
        message = []
        message_compact = []
        for i in range(self.n_messages):

            assert len(message_lst[i]) == self.config.message_len - 1
            index = np.concatenate((np.array(message_lst[i])+1, [0]))
            one_hot = np.identity(self.message_dim)[index]
            message_compact.append(one_hot)
            if patch_len % self.message_len == 0:
                message.append(np.tile(one_hot.T, (1, patch_len // self.message_len)))
            else:
                _ = np.tile(one_hot.T, (1, patch_len // self.message_len))
                _ = np.concatenate([_, one_hot.T[:, 0:patch_len % self.message_len]], axis=1)
                message.append(_)
        message = np.stack(message)
        message_compact = np.stack(message_compact)
        # message = np.pad(message, ((0, 0), (0, 129 - self.message_dim), (0, 0)), 'constant')
        return message, message_compact
    
    def get_best_ps(self, y_one_sec):
        
        def check_accuracy(pred_values):
        
            accuracy = 0
            for i in range(pred_values.shape[1]):
                unique, counts = np.unique(pred_values[:, i], return_counts=True)
                accuracy += np.max(counts) / pred_values.shape[0]
            
            return accuracy / pred_values.shape[1]

        y = torch.FloatTensor(y_one_sec).unsqueeze(0).unsqueeze(0).to(self.device)
        max_accuracy = 0
        final_phase_shift = 0

        for ps in range(0, self.config.HOP_LENGTH, 10):

            carrier, _ = self.stft.transform(y[0:1, 0:1, ps:].squeeze(1))
            carrier = carrier[:, None]

            for i in range(self.n_messages):  # decode each msg_i using decoder_m_i
                msg_reconst = self.dec_m[i](carrier)
                # soft = torch.nn.functional.softmax(msg_reconst, dim=2)
                # max_entropy = np.ones([self.message_dim])/self.message_dim
                # print(ps, -torch.mean(torch.sum(soft*torch.log(soft), dim=2)), -np.sum(max_entropy*np.log(max_entropy)))
                pred_values = torch.argmax(msg_reconst[0, 0], dim=0).data.cpu().numpy()
                pred_values = pred_values[0:int(msg_reconst.shape[3]/self.config.message_len)*self.config.message_len]
                pred_values = pred_values.reshape([-1, self.config.message_len])
                # print(pred_values, i)
                # if i == 0:
                    # return 0
                cur_acc = check_accuracy(pred_values)
                if cur_acc > max_accuracy:
                    max_accuracy = cur_acc
                    final_phase_shift = ps

        return final_phase_shift
    
    def get_confidence(self, pred_values, message):
        
        assert len(message) == pred_values.shape[1], f'{len(message)} | {pred_values.shape}'
        return np.mean((pred_values == message[None]).astype(np.float32)).item()
    
    def sdr(self, orig, recon):
        rms1 = ((np.mean(orig ** 2)) ** 0.5)
        rms2 = ((np.mean((orig - recon) ** 2)) ** 0.5)
        sdr = 20 * np.log10(rms1 / rms2)
        return sdr

    def load_audio(self, path):

        # return librosa.load(path, sr=None)

        audio = AudioSegment.from_file(path)
        return (np.array(audio.get_array_of_samples(), dtype=np.float32).reshape((-1, audio.channels)) / (
            1 << (8 * audio.sample_width - 1)))[:, 0], audio.frame_rate, audio.channels

    def encode(self, in_path, out_path, message, message_sdr=None):

        if message_sdr is None:
            message_sdr = self.config.message_sdr
            print(f'Using the default SDR of {self.config.message_sdr} dB')

        with torch.no_grad():

            start = time.time()

            y, orig_sr, channels = self.load_audio(in_path)
            if channels != 1:
                return {'status': False, 'message': 'Currently only supporting single channel audio'}
            orig_y = y.copy()
            if orig_sr != self.sr:
                if orig_sr > self.sr:
                    print(f'WARNING! Reducing the sampling rate of the original audio from {orig_sr} -> {self.sr}. High frequency components may be lost!')
                y = librosa.resample(y, orig_sr = orig_sr, target_sr = self.sr)
            original_power = np.mean(y**2)

            y = y * np.sqrt(self.average_energy_VCTK / original_power)  # Noise has a power of 5% power of VCTK samples
            y = torch.FloatTensor(y).unsqueeze(0).unsqueeze(0).to(self.device)
            carrier, carrier_phase = self.stft.transform(y.squeeze(1))
            carrier = carrier[:, None]
            carrier_phase = carrier_phase[:, None]

            def binary_encode(mes):
                binary_message = ''.join(['{0:08b}'.format(mes_i) for mes_i in mes])
                four_bit_msg = []
                for i in range(len(binary_message)//2):
                    four_bit_msg.append(int(binary_message[i*2:i*2+2], 2))
                return four_bit_msg
            
            binary_encoded_message = binary_encode(message)
            # print(binary_encoded_message)

            msgs, msgs_compact = self.letters_encoding(carrier.shape[3], [binary_encoded_message])
            msg_enc = torch.from_numpy(msgs[None]).to(self.device).float()

            carrier_enc = self.enc_c(carrier)  # encode the carrier
            # print(carrier.shape, carrier_enc.shape)  # torch.Size([8, 1, 129, 1036]) torch.Size([8, 32, 129, 1036])
            if self.device == 'cpu':
                msg_enc = self.enc_c.transform_message(msg_enc)
            else:
                msg_enc = self.enc_c.module.transform_message(msg_enc)

            # print(carrier_enc.shape, carrier.shape, msg_enc.shape)
            merged_enc = torch.cat((carrier_enc, carrier.repeat(1, 32, 1, 1), msg_enc.repeat(1, 32, 1, 1)), dim=1)  # concat encodings on features axis
            
            message_info = self.dec_c(merged_enc, message_sdr)
            if self.config.frame_level_normalization:
                message_info = message_info*(torch.mean((carrier**2), dim=2, keepdim=True)**0.5)  # *time_weighing
            elif self.config.utterance_level_normalization:
                message_info = message_info*(torch.mean((carrier**2), dim=(2,3), keepdim=True)**0.5)  # *time_weighing
            
            if self.config.ensure_negative_message:
                message_info = -message_info
                carrier_reconst = torch.nn.functional.relu(message_info + carrier)  # decode carrier, output in stft domain
            elif self.config.ensure_constrained_message:
                message_info[message_info > carrier] = carrier[message_info > carrier]
                message_info[-message_info > carrier] = -carrier[-message_info > carrier]
                carrier_reconst = message_info + carrier  # decode carrier, output in stft domain
                assert torch.all(carrier_reconst >= 0), 'negative values found in carrier_reconst'
            else:
                carrier_reconst = torch.abs(message_info + carrier)  # decode carrier, output in stft domain

            self.stft.num_samples = y.shape[2]

            y = self.stft.inverse(carrier_reconst.squeeze(1), carrier_phase.squeeze(1)).data.cpu().numpy()[0, 0]
            y = y * np.sqrt(original_power / (self.average_energy_VCTK))  # Noise has a power of 5% power of VCTK samples
            if orig_sr != self.sr:
                y = librosa.resample(y, orig_sr = self.sr, target_sr = orig_sr)

            # print(y.shape, orig_y.shape, self.stft.num_samples, orig_sr, self.sr)
            time_taken = time.time() - start
            # print('Time taken to encode', time_taken)
            # print('Time taken per second of audio to encode', time_taken / (y.shape[0] / orig_sr))
            sdr = self.sdr(orig_y, y)
            sf.write(out_path, y, orig_sr)
            
        return {'status': True, 'sdr': f'{sdr:.2f}', 'time_taken': time_taken, 'time_taken_per_second': time_taken / (y.shape[0] / orig_sr)}
    
    def decode(self, path, phase_shift_decoding):
        try:
            with torch.no_grad():
                y, orig_sr, channels = self.load_audio(path)
                if channels != 1:
                    return {'status': False, 'message': 'Currently only supporting single channel audio'}
                if orig_sr != self.sr:
                    y = librosa.resample(y, orig_sr = orig_sr, target_sr = self.sr)
                original_power = np.mean(y**2)
                y = y * np.sqrt(self.average_energy_VCTK / original_power)  # Noise has a power of 5% power of VCTK samples
                if phase_shift_decoding and phase_shift_decoding != 'false':
                    ps = self.get_best_ps(y)
                else:
                    ps = 0
                y = torch.FloatTensor(y[ps:]).unsqueeze(0).unsqueeze(0).to(self.device)
                carrier, _ = self.stft.transform(y.squeeze(1))
                carrier = carrier[:, None]

                msg_reconst_list = []
                confidence = []

                for i in range(self.n_messages):  # decode each msg_i using decoder_m_i
                    msg_reconst = self.dec_m[i](carrier)
                    pred_values = torch.argmax(msg_reconst[0, 0], dim=0).data.cpu().numpy()
                    pred_values = pred_values[0:int(msg_reconst.shape[3]/self.config.message_len)*self.config.message_len]
                    pred_values = pred_values.reshape([-1, self.config.message_len])

                    ord_values = st.mode(pred_values, keepdims=False).mode
                    end_char = np.min(np.nonzero(ord_values == 0)[0])
                    confidence.append(self.get_confidence(pred_values, ord_values))
                    if end_char == self.config.message_len:
                        ord_values = ord_values[:self.config.message_len-1]
                    else:
                        ord_values = np.concatenate([ord_values[end_char+1:], ord_values[:end_char]], axis=0)

                    # pred_values = ''.join([chr(v + 64) for v in ord_values])
                    msg_reconst_list.append((ord_values - 1).tolist())
                
                def convert_to_8_bit_segments(msg_list):
                    segment_message_list = []
                    for msg_list_i in msg_list:
                        binary_format = ''.join(['{0:02b}'.format(mes_i) for mes_i in msg_list_i])
                        eight_bit_segments = [int(binary_format[i*8:i*8+8], 2) for i in range(len(binary_format)//8)]
                        segment_message_list.append(eight_bit_segments)
                    return segment_message_list
                msg_reconst_list = convert_to_8_bit_segments(msg_reconst_list)
            return {'messages': msg_reconst_list, 'confidences': confidence, 'status': True}
        except:
            return {'messages': [], 'confidences': [], 'error': 'Could not find message', 'status': False}
    
    def encode_wav(self, y, orig_sr, message, message_sdr=None):

        if message_sdr is None:
            message_sdr = self.config.message_sdr
            print(f'Using the default SDR of {self.config.message_sdr} dB')

        with torch.no_grad():

            start = time.time()
        
            orig_y = y.copy()
            if orig_sr != self.sr:
                if orig_sr > self.sr:
                    print(f'WARNING! Reducing the sampling rate of the original audio from {orig_sr} -> {self.sr}. High frequency components may be lost!')
                y = librosa.resample(y, orig_sr = orig_sr, target_sr = self.sr)
            original_power = np.mean(y**2)

            y = y * np.sqrt(self.average_energy_VCTK / original_power)  # Noise has a power of 5% power of VCTK samples
            y = torch.FloatTensor(y).unsqueeze(0).unsqueeze(0).to(self.device)
            carrier, carrier_phase = self.stft.transform(y.squeeze(1))
            carrier = carrier[:, None]
            carrier_phase = carrier_phase[:, None]

            def binary_encode(mes):
                binary_message = ''.join(['{0:08b}'.format(mes_i) for mes_i in mes])
                four_bit_msg = []
                for i in range(len(binary_message)//2):
                    four_bit_msg.append(int(binary_message[i*2:i*2+2], 2))
                return four_bit_msg
            
            binary_encoded_message = binary_encode(message)
            # print(binary_encoded_message)

            msgs, msgs_compact = self.letters_encoding(carrier.shape[3], [binary_encoded_message])
            msg_enc = torch.from_numpy(msgs[None]).to(self.device).float()

            carrier_enc = self.enc_c(carrier)  # encode the carrier
            # print(carrier.shape, carrier_enc.shape)  # torch.Size([8, 1, 129, 1036]) torch.Size([8, 32, 129, 1036])
            if self.device == 'cpu':
                msg_enc = self.enc_c.transform_message(msg_enc)
            else:
                msg_enc = self.enc_c.module.transform_message(msg_enc)

            # print(carrier_enc.shape, carrier.shape, msg_enc.shape)
            merged_enc = torch.cat((carrier_enc, carrier.repeat(1, 32, 1, 1), msg_enc.repeat(1, 32, 1, 1)), dim=1)  # concat encodings on features axis
            
            message_info = self.dec_c(merged_enc, message_sdr)
            if self.config.frame_level_normalization:
                message_info = message_info*(torch.mean((carrier**2), dim=2, keepdim=True)**0.5)  # *time_weighing
            elif self.config.utterance_level_normalization:
                message_info = message_info*(torch.mean((carrier**2), dim=(2,3), keepdim=True)**0.5)  # *time_weighing
            
            if self.config.ensure_negative_message:
                message_info = -message_info
                carrier_reconst = torch.nn.functional.relu(message_info + carrier)  # decode carrier, output in stft domain
            elif self.config.ensure_constrained_message:
                message_info[message_info > carrier] = carrier[message_info > carrier]
                message_info[-message_info > carrier] = -carrier[-message_info > carrier]
                carrier_reconst = message_info + carrier  # decode carrier, output in stft domain
                assert torch.all(carrier_reconst >= 0), 'negative values found in carrier_reconst'
            else:
                carrier_reconst = torch.abs(message_info + carrier)  # decode carrier, output in stft domain

            self.stft.num_samples = y.shape[2]

            y = self.stft.inverse(carrier_reconst.squeeze(1), carrier_phase.squeeze(1)).data.cpu().numpy()[0, 0]
            y = y * np.sqrt(original_power / (self.average_energy_VCTK))  # Noise has a power of 5% power of VCTK samples
            if orig_sr != self.sr:
                y = librosa.resample(y, orig_sr = self.sr, target_sr = orig_sr)

            # print(y.shape, orig_y.shape, self.stft.num_samples, orig_sr, self.sr)
            time_taken = time.time() - start
            # print('Time taken to encode', time_taken)
            # print('Time taken per second of audio to encode', time_taken / (y.shape[0] / orig_sr))
            sdr = self.sdr(orig_y, y)
            
        return y, sdr
    
    def decode_wav(self, y, orig_sr, phase_shift_decoding):
        # print(path, phase_shift_decoding)
        try:
            with torch.no_grad():
                if orig_sr != self.sr:
                    y = librosa.resample(y, orig_sr = orig_sr, target_sr = self.sr)
                original_power = np.mean(y**2)
                y = y * np.sqrt(self.average_energy_VCTK / original_power)  # Noise has a power of 5% power of VCTK samples
                if phase_shift_decoding and phase_shift_decoding != 'false':
                    ps = self.get_best_ps(y)
                else:
                    ps = 0
                y = torch.FloatTensor(y[ps:]).unsqueeze(0).unsqueeze(0).to(self.device)
                carrier, _ = self.stft.transform(y.squeeze(1))
                carrier = carrier[:, None]

                msg_reconst_list = []
                confidence = []

                for i in range(self.n_messages):  # decode each msg_i using decoder_m_i
                    msg_reconst = self.dec_m[i](carrier)
                    pred_values = torch.argmax(msg_reconst[0, 0], dim=0).data.cpu().numpy()
                    pred_values = pred_values[0:int(msg_reconst.shape[3]/self.config.message_len)*self.config.message_len]
                    pred_values = pred_values.reshape([-1, self.config.message_len])

                    ord_values = st.mode(pred_values, keepdims=False).mode
                    end_char = np.min(np.nonzero(ord_values == 0)[0])
                    confidence.append(self.get_confidence(pred_values, ord_values))
                    if end_char == self.config.message_len:
                        ord_values = ord_values[:self.config.message_len-1]
                    else:
                        ord_values = np.concatenate([ord_values[end_char+1:], ord_values[:end_char]], axis=0)

                    # pred_values = ''.join([chr(v + 64) for v in ord_values])
                    msg_reconst_list.append((ord_values - 1).tolist())
                
                def convert_to_8_bit_segments(msg_list):
                    segment_message_list = []
                    for msg_list_i in msg_list:
                        binary_format = ''.join(['{0:02b}'.format(mes_i) for mes_i in msg_list_i])
                        eight_bit_segments = [int(binary_format[i*8:i*8+8], 2) for i in range(len(binary_format)//8)]
                        segment_message_list.append(eight_bit_segments)
                    return segment_message_list
                msg_reconst_list = convert_to_8_bit_segments(msg_reconst_list)
            return {'messages': msg_reconst_list, 'confidences': confidence, 'status': True}
        except:
            return {'messages': [], 'confidences': [], 'error': 'Could not find message', 'status': False}
    
    def convert_dataparallel_to_normal(self, dictionary):

        return {i[len('module.'):]: dictionary[i] for i in dictionary}

    def load_models(self, ckpt_dir):

        if self.device == 'cpu':
            self.enc_c.load_state_dict(self.convert_dataparallel_to_normal(torch.load(os.path.join(ckpt_dir, "enc_c.ckpt"), map_location=self.device)))
            self.dec_c.load_state_dict(self.convert_dataparallel_to_normal(torch.load(os.path.join(ckpt_dir, "dec_c.ckpt"), map_location=self.device)))
            for i,m in enumerate(self.dec_m):
                m.load_state_dict(self.convert_dataparallel_to_normal(torch.load(os.path.join(ckpt_dir, f"dec_m_{i}.ckpt"), map_location=self.device)))
        else:
            self.enc_c.load_state_dict(torch.load(os.path.join(ckpt_dir, "enc_c.ckpt"), map_location=self.device))
            self.dec_c.load_state_dict(torch.load(os.path.join(ckpt_dir, "dec_c.ckpt"), map_location=self.device))
            for i,m in enumerate(self.dec_m):
                m.load_state_dict(torch.load(os.path.join(ckpt_dir, f"dec_m_{i}.ckpt"), map_location=self.device))


def get_model(model_type='44.1k', ckpt_path='../Models/44_1_khz/73999_iteration', config_path='../Models/44_1_khz/73999_iteration/hparams.yaml', device='cpu'):

    if model_type == '44.1k':
        config = yaml.safe_load(open(config_path))
        config = argparse.Namespace(**config)
        config.load_ckpt = ckpt_path
        model = Model(config, device)
    elif model_type == '16k':
        config = yaml.safe_load(open(config_path))
        config = argparse.Namespace(**config)
        config.load_ckpt = ckpt_path

        model = Model(config, device)
    else:
        print('Please specify a valid model_type [44.1k, 16k]')
    
    return model
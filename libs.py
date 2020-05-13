#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pyaudio
import wave
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy.signal import hanning, welch

class AudioVisualization(object):

    def __init__(self, audio):
        self.audio = audio
        self.fig, self.ax = plt.subplots(3, figsize=(6.4,4.8))

        xt = np.arange(0, audio.chunk)
        yt = np.zeros(audio.chunk)
        xf, yf = self.fft(audio)
        xp, yp = self.psd(audio)

        # ax[0] - time domain
        self.line_t, = self.ax[0].plot(xt, yt, 'b-', lw=2)
        self.ax[0].set_title('Audio Waveform')
        self.ax[0].set_xlabel('samples')
        self.ax[0].set_ylabel('volume')
        self.ax[0].set_xlim(0, audio.chunk)
        self.ax[0].set_ylim(-32768, 32768)

        # ax[1] - fft
        self.line_fft, = self.ax[1].plot(xf, yf, 'r-', lw=1)
        self.ax[1].set_title('FFT')
        self.ax[1].set_xlabel('frequency')
        self.ax[1].set_ylabel('Amplitude')
        self.ax[1].set_xlim(20, audio.sample_rate/2)
        self.ax[1].set_ylim(0, 100)

        # ax[2] - psd
        self.line_psd, = self.ax[2].plot(xp, yp, 'r-', lw=1)
        self.ax[2].set_title('PSD')
        self.ax[2].set_xlabel('frequency')
        self.ax[2].set_ylabel('dB/Hz')
        self.ax[2].set_xlim(20, audio.sample_rate/2)
        self.ax[2].set_ylim(-50, 50)

    def update(self):
        # time domain
        self.line_t.set_ydata(self.audio.get_int_data())

        # frequency domain
        _, yf = self.fft(self.audio)
        self.line_fft.set_ydata(yf)

        # psd
        _, yp = self.psd(self.audio)
        self.line_psd.set_ydata(yp)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        return self.fig

    def show(self):
        plt.show(block=False)
        try:
            while self.audio.stream.is_active():
                self.update()
            self.audio.stop()
        except KeyboardInterrupt:
            self.audio.stop()

    def stop(self):
        plt.close('all')

    @staticmethod
    def fft(audio):
        f = fftpack.fftfreq(audio.chunk, 1.0/audio.sample_rate)[:audio.chunk//2]
        amp = fftpack.fft(audio.window * audio.get_int_data())
        amp = np.abs(amp[:audio.chunk//2]) * 2 / (audio.chunk)
        return f, amp

    @staticmethod
    def psd(audio):
        f, Pxxf = welch(
            audio.get_int_data(),
            audio.sample_rate,
            nperseg=audio.chunk,
            return_onesided=True,
            detrend=False)
        dbhz = 10*np.log10(Pxxf)
        return f, dbhz


class AudioStreamBasic(object):

    def __init__(self, audio_format, sample_rate, channels, chunk, audio_input, audio_output):
        self.pyaudio = pyaudio.PyAudio()

        self.format = audio_format
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk = chunk

        self.input = audio_input
        if self.input:
            self.input_device_index = self.pyaudio.get_default_input_device_info()['index']
        else:
            self.input_device_index = None

        self.output = audio_output
        if self.output:
            self.output_device_index = self.pyaudio.get_default_output_device_info()['index']
        else:
            self.output_device_index = None

        self.window = hanning(self.chunk, True)

        self.stream = self.pyaudio.open(
            rate=self.sample_rate,
            format=self.format,
            channels=self.channels,
            input=self.input,
            input_device_index=self.input_device_index,
            output=self.output,
            output_device_index=self.output_device_index,
            frames_per_buffer=self.chunk,
            stream_callback=self.callback)
        self._data = None

    @staticmethod
    def str_to_int(data):
        return np.fromstring(data, dtype=np.int16)

    def get_raw_data(self):
        return self._data

    def get_int_data(self):
        data = self.get_raw_data()
        return self.str_to_int(data)

    def callback(self, in_data, frame_count, time_info, status):
        return (in_data, pyaudio.paContinue)

    def start(self):
        self.stream.start_stream()

    def stop(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pyaudio.terminate()


class AudioStreamFile(AudioStreamBasic):

    def __init__(self, filename, chunk=1024):
        self.filename = filename
        self.wavfile = wave.open(self.filename, 'rb')

        audio_format = pyaudio.get_format_from_width(self.wavfile.getsampwidth())
        sample_rate = self.wavfile.getframerate()
        channels = self.wavfile.getnchannels()

        super(AudioStreamFile, self).__init__(audio_format, sample_rate, channels, chunk, audio_input=False, audio_output=True)

    def callback(self, in_data, frame_count, time_info, status):
        data = self.wavfile.readframes(frame_count)
        self._data = data
        return (data, pyaudio.paContinue)


class AudioStreamLive(AudioStreamBasic):

    def __init__(self, sample_rate=44100, channels=1, chunk=1024, audio_output=False, save_folder='', save_interval=10):
        audio_format = pyaudio.paInt16
        super(AudioStreamLive, self).__init__(audio_format, sample_rate, channels, chunk, audio_input=True, audio_output=audio_output)
        self.save_folder = save_folder
        self.save_interval = save_interval
        self._rec_flag = False
        self._records = []
        self._rec_timer = time.time()

    def set_rec(self, state):
        if state == self._rec_flag:
            return
        if state:
            print('audio recording...')
            self._rec_timer = time.time()
        else:
            print('stop recording.')
        self._rec_flag = state

    def save_wav(self):
        fn = os.path.join(self.save_folder, 'record_' + time.strftime('%Y%m%d_%H%M%S') + '.wav')
        wavfile = wave.open(fn, 'wb')
        wavfile.setnchannels(self.channels)
        wavfile.setsampwidth(self.pyaudio.get_sample_size(self.format))
        wavfile.setframerate(self.sample_rate)
        wavfile.writeframes(b''.join(self._records))
        wavfile.close()
        print('save audio as {}'.format(fn))

    def callback(self, in_data, frame_count, time_info, status):
        self._data = in_data

        if self._rec_flag:
            self._records.append(in_data)
        # else:
            if self._records and time.time() - self._rec_timer > self.save_interval:
                self.save_wav()
                self._records = []
                self._rec_timer = time.time()
        return (in_data, pyaudio.paContinue)

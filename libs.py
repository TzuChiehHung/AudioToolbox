#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pyaudio
import wave
import numpy as np
import RPi.GPIO as GPIO
import time
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy.signal import hanning, welch


class AudioVisualization(object):

    def __init__(self, audio):
        self.audio = audio
        self.fig, self.ax = plt.subplots(3, figsize=(8,6))

        xt = np.arange(0, audio.chunk)
        yt = np.zeros(audio.chunk)
        xf, yf = fft(audio)
        xp, yp = psd(audio)

        # ax[0] - time domain
        self.line_t, = self.ax[0].plot(xt, yt, 'b-', lw=2)
        self.ax[0].set_title('Audio Waveform')
        self.ax[0].set_xlabel('samples')
        self.ax[0].set_ylabel('volume')
        self.ax[0].set_xlim(0, audio.chunk)
        self.ax[0].set_ylim(-1, 1)

        # ax[1] - fft
        self.line_fft, = self.ax[1].plot(xf, yf, 'r-', lw=1)
        self.ax[1].set_title('FFT')
        self.ax[1].set_xlabel('frequency')
        self.ax[1].set_ylabel('Amplitude')
        self.ax[1].set_xlim(20, audio.sample_rate/2)
        self.ax[1].set_ylim(0, 0.01)

        # ax[2] - psd
        self.line_psd, = self.ax[2].plot(xp, yp, 'r-', lw=1)
        self.ax[2].set_title('PSD')
        self.ax[2].set_xlabel('frequency')
        self.ax[2].set_ylabel('dB/Hz')
        self.ax[2].set_xlim(20, audio.sample_rate/2)
        self.ax[2].set_ylim(-150, -10)

    def show(self):
        plt.show(block=False)

        try:
            while self.audio.stream.is_active():
                # time domain
                self.line_t.set_ydata(self.audio.data)

                # frequency domain
                _, yf = fft(self.audio)
                self.line_fft.set_ydata(yf)

                # psd
                _, yp = psd(self.audio)
                self.line_psd.set_ydata(yp)

                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
            self.audio.stop()
        except KeyboardInterrupt:
            self.audio.stop()

class AudioFileStream(object):

    def __init__(self, filename, chunk=1024, output_device=None):
        self.filename = filename
        self.wavfile = wave.open(self.filename, 'rb')

        self.pyaudio = pyaudio.PyAudio()

        self.format = self.pyaudio.get_format_from_width(self.wavfile.getsampwidth())
        self.sample_rate = self.wavfile.getframerate()
        self.channels = self.wavfile.getnchannels()
        self.chunk = chunk
        self.window = hanning(self.chunk, True)

        if output_device is None:
            self.output_device_index = self.pyaudio.get_default_output_device_info()['index']
        else:
            self.output_device_index = output_device

        self.stream = self.pyaudio.open(
            rate=self.sample_rate,
            format=self.format,
            channels=self.channels,
            output=True,
            output_device_index=self.output_device_index,
            frames_per_buffer=self.chunk*2,
            stream_callback=self.callback)
        self.data = np.zeros(self.chunk)

    def callback(self, in_data, frame_count, time_info, status):
        data = self.wavfile.readframes(frame_count)
        self.data = self.str_to_float(data)
        return (data, pyaudio.paContinue)

    def str_to_float(self, data):
        MAX_INT = 32768.0
        return np.fromstring(data, dtype=np.int16) / MAX_INT

    def start(self):
        self.stream.start_stream()

    def stop(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pyaudio.terminate()

class AudioLiveStream(object):

    def __init__(self, sample_rate=44100, channels=1, chunk=1024, input_device=None, output_device=None):
        self.format = pyaudio.paInt16

        self.button = 17
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.button, GPIO.IN)
        self.btn_state = GPIO.input(self.button)

        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk = chunk
        self.window = hanning(self.chunk, True)

        self.pyaudio = pyaudio.PyAudio()
        if input_device is None:
            self.input_device_index = self.pyaudio.get_default_input_device_info()['index']
        else:
            self.input_device_index = input_device
        if output_device is None:
            self.output_device_index = self.pyaudio.get_default_output_device_info()['index']
        else:
            self.output_device_index = output_device
        self.stream = self.pyaudio.open(
            rate=self.sample_rate,
            format=self.format,
            channels=self.channels,
            input=True,
            input_device_index=self.input_device_index,
            output=True,
            output_device_index=self.output_device_index,
            frames_per_buffer=self.chunk,
            stream_callback=self.callback)
        self.data = np.zeros(self.chunk)
        self.records = []

    def callback(self, in_data, frame_count, time_info, status):
        self.data = self.str_to_float(in_data)
        self.btn_state = GPIO.input(self.button)
        if self.btn_state:
            if self.records:
                timestr = time.strftime('%Y%m%d_%H%M%S') + '.wav'
                wavfile = wave.open(timestr, 'wb')
                wavfile.setnchannels(self.channels)
                wavfile.setsampwidth(self.pyaudio.get_sample_size(self.format))
                wavfile.setframerate(self.sample_rate)
                wavfile.writeframes(b''.join(self.records))
                wavfile.close()
                print('save audio as {}'.format(timestr))
                self.records = []
        else:
            self.records.append(in_data)
        return (in_data, pyaudio.paContinue)

    def str_to_float(self, data):
        MAX_INT = 32768.0
        return np.fromstring(data, dtype=np.int16) / MAX_INT

    def start(self):
        self.stream.start_stream()

    def stop(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pyaudio.terminate()

def fft(audio):
    f = fftpack.fftfreq(audio.chunk, 1.0/audio.sample_rate)[:audio.chunk//2]
    amp = fftpack.fft(audio.window * audio.data)
    amp = np.abs(amp[:audio.chunk//2]) * 2 / (audio.chunk)
    return f, amp

def psd(audio):
    f, Pxxf = welch(
        audio.data,
        audio.sample_rate,
        nperseg=audio.chunk,
        return_onesided=True, 
        detrend=False)
    dbhz = 10*np.log10(Pxxf)
    return f, dbhz
